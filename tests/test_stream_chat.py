"""Phase 2 UI + pipeline integration tests, with mocked local model and index."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from src.chat_ui import (
    WAITING, DRAFT_PREFIX, apply_stream_event, cancel_turn,
    append_error, start_turn, sources_message, stream_chat_turn, AUTO_LABEL,
)
from src.rag_pipeline import LocalDrugRAG
from src.prompt_builder import ABSTENTION

PDF = "ibuprofen_200mg_haleon_us_holdings_llc.pdf"
URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=xyz"
QUESTION = "What safety warnings are listed for ibuprofen?"


def match():
    text = "The label warns that ibuprofen can cause stomach bleeding."
    return {"rank":1,"chunk_id":"a1","text":text,"filename":PDF,
            "page_number":2,"chunk_index":1,"start_char":0,"end_char":len(text),
            "source_url":URL,"cosine_distance":.1,"cosine_similarity":.9}


class StreamChatTests(TestCase):
    def test_draft_updates_existing_reply_not_new_bubble(self):
        _, past, _ = start_turn(QUESTION, [])
        h1 = apply_stream_event(past, {"type":"status","message":"Searching..."})
        h2 = apply_stream_event(h1, {"type":"delta","text":"Bleeding "})
        h3 = apply_stream_event(h2, {"type":"delta","text":"Bleeding risk [S1]"})
        self.assertEqual(len(h3), 2)
        self.assertIn("not citation-verified", h3[-1]["content"])
        self.assertTrue(h3[-1]["content"].endswith("Bleeding risk [S1]"))
        self.assertEqual(past[-1]["content"], WAITING)

    def test_final_replaces_unverified_and_shows_sources(self):
        _, history, _ = start_turn(QUESTION, [])
        history = apply_stream_event(history,{"type":"delta","text":"Unverified wrong claim"})
        result = {"status":"answered","answer":"Bleeding risk [S1].","sources":[{
            "citation":"[S1]","page":2,"excerpt":"Bleeding risk.",
            "source_url":URL,"used_in_answer":True,
        }]}
        final = apply_stream_event(history,{"type":"final","result":result})
        self.assertNotIn("Unverified wrong claim", repr(final))
        self.assertNotIn(DRAFT_PREFIX, final[1]["content"])
        self.assertEqual(final[1]["content"], result["answer"])
        self.assertEqual(len(final),3)

    def test_rejected_stream_discards_all_draft_claims(self):
        _, history, _ = start_turn(QUESTION, [])
        history = apply_stream_event(history,{"type":"delta","text":"Invented dosage recommendation"})
        final = apply_stream_event(history,{"type":"final","result":{
            "status":"rejected_unknown_citation","answer":ABSTENTION,
            "reason":"Invalid citation", "sources":[],
        }})
        self.assertNotIn("Invented dosage", repr(final))
        self.assertIn(ABSTENTION, final[-1]["content"])

    def test_error_scrubs_unvalidated_text(self):
        _, history, _ = start_turn(QUESTION, [])
        history = apply_stream_event(history,{"type":"delta","text":"Invented medical advice"})
        final = append_error(history, RuntimeError("internal error"))
        self.assertNotIn("Invented medical advice", repr(final))
        self.assertIn("Unable to answer safely", final[-1]["content"])

    def test_user_cancel_scrubs_unverified_text(self):
        _, history, _ = start_turn(QUESTION, [])
        history = apply_stream_event(history,{"type":"delta","text":"Draft"})
        final = cancel_turn(history)
        self.assertNotIn("Draft", repr(final))
        self.assertIn("cancelled", final[-1]["content"])
        self.assertEqual(cancel_turn(final), final)

    def test_no_duplicate_submissions_while_streaming(self):
        _, history, _ = start_turn(QUESTION, [])
        unsubmitted, same_history, pending = start_turn("Another question", history)
        self.assertEqual(unsubmitted, "Another question")
        self.assertEqual(pending, "")
        self.assertEqual(same_history, history)

    def test_invalid_stream_events_fail(self):
        _, h, _ = start_turn(QUESTION, [])
        for event in (None, {"type":"delta","text":None},
                      {"type":"final","result":None},{"type":"nonsense"}):
            with self.subTest(event=event), self.assertRaises(RuntimeError):
                apply_stream_event(h,event)

    def test_no_pending_turn_must_not_be_overwritten(self):
        with self.assertRaisesRegex(RuntimeError,"No pending"):
            apply_stream_event([{"role":"assistant","content":"validated response"}],
                               {"type":"delta","text":"unverified"})

    def test_stream_pipeline_yields_incremental_then_cited_final(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data').mkdir()
            (root/'data/sources.csv').write_text('filename\n'+PDF+'\n',encoding='utf-8')
            (root/'data/target_drugs.txt').write_text('ibuprofen\n',encoding='utf-8')
            called=[]
            def streamed(package, **kwargs):
                called.append(package)
                yield {"type":"delta","text":"Bleeding "}
                yield {"type":"delta","text":"Bleeding risk [S1]."}
                yield {"type":"final","result":{
                    "status":"answered","answer":"Bleeding risk [S1].",
                    "reason":"IDs valid","citations":["S1"],
                    "usage":{"prompt_eval_count":800,"eval_count":4},
                }}
            svc=LocalDrugRAG(root,collection=Mock(count=lambda:7194),embedder=Mock(),stream_generator=streamed)
            with patch('src.rag_pipeline.retrieve_chunks',return_value=[match()]) as retr:
                events=list(svc.ask_stream(QUESTION))
            self.assertEqual([e['type'] for e in events], ['status','status','delta','delta','final'])
            self.assertEqual(events[-1]['result']['status'],'answered')
            self.assertEqual(events[-1]['result']['sources'][0]['citation'],'[S1]')
            retr.assert_called_once()
            self.assertEqual(len(called),1)

    def test_stream_pipeline_no_scope_skips_transport(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data').mkdir()
            (root/'data/sources.csv').write_text('filename\n'+PDF+'\n',encoding='utf-8')
            (root/'data/target_drugs.txt').write_text('ibuprofen\n',encoding='utf-8')
            svc=LocalDrugRAG(root,stream_generator=lambda *a,**k:self.fail('called Ollama'))
            events=list(svc.ask_stream('What are warnings?'))
            self.assertEqual([x['type'] for x in events],['status','final'])
            self.assertEqual(events[-1]['result']['status'],'needs_drug_scope')

    def test_actual_ui_callback_logic_emits_live_updates(self):
        _, history, pending = start_turn(QUESTION, [])
        class FakeService:
            sources_csv = None
            targets_txt = None
            def ask_stream(self, question, **kwargs):
                yield {"type":"status","message":"Searching"}
                yield {"type":"delta","text":"Some "}
                yield {"type":"delta","text":"Some text [S1]"}
                yield {"type":"final","result":{"status":"answered",
                    "answer":"Some text [S1]", "sources":[]}}
        outs=list(stream_chat_turn(history,pending,AUTO_LABEL,3,FakeService()))
        self.assertEqual(len(outs),4)
        self.assertTrue(outs[1][-1]['content'].startswith(DRAFT_PREFIX))
        self.assertEqual(outs[2][-1]['content'], DRAFT_PREFIX + 'Some text [S1]')
        self.assertEqual(outs[3][-1]['content'],'Some text [S1]')

    def test_actual_ui_callback_logic_scrubs_midstream_failure(self):
        _, history, pending = start_turn(QUESTION, [])
        class FailingService:
            sources_csv = None
            targets_txt = None
            def ask_stream(self, question, **kwargs):
                yield {"type":"delta","text":"Wrong medical advice"}
                raise RuntimeError('Ollama OOM')
        outs=list(stream_chat_turn(history,pending,AUTO_LABEL,3,FailingService()))
        self.assertEqual(len(outs),2)
        self.assertNotIn('Wrong medical advice', repr(outs[-1]))
        self.assertIn('Unable to answer safely', outs[-1][-1]['content'])

    def test_ui_loads_without_ollama_or_gpu(self):
        if importlib.util.find_spec('gradio') is None:
            self.skipTest('Optional Gradio smoke test')
        from scripts.app import build_app
        with TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data').mkdir()
            (root/'data/sources.csv').write_text('filename\n'+PDF+'\n',encoding='utf-8')
            (root/'data/target_drugs.txt').write_text('ibuprofen\n',encoding='utf-8')
            svc=LocalDrugRAG(root, collection=Mock(), embedder=Mock())
            app=build_app(service=svc)
            self.assertIsNotNone(app)
            app.close()
