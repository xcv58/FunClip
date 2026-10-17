import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from funclip.glossary import (
    GlossaryError, glossary_context, glossary_report, parse_glossary,
    recognition_hotwords, resolve_glossary, validate_glossary_correction,
    validate_glossary_translation,
)
from funclip.llm import srt_corrector, srt_translator


def srt(*texts):
    return '\n\n'.join(f'{i}\n00:00:0{i-1},000 --> 00:00:0{i},000\n{text}'
                       for i, text in enumerate(texts, 1))


def response(kind, *texts):
    payload = {kind: [{'cue_id': i, 'text': text} for i, text in enumerate(texts, 1)]}
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
        content=json.dumps(payload, ensure_ascii=False), refusal=None))], usage=None)


class NativeParser:
    def __init__(self):
        self.files = []

    def generate_hotwords_list(self, path, **kwargs):
        self.files.append(path)
        words = Path(path).read_text().splitlines()
        return [[0 if any(c in word for c in '蟳𫊻') else 1] for word in words] + [[2]]


def native_model():
    return SimpleNamespace(model=NativeParser(), kwargs={
        'token_list': ['<unk>', 'known', '<s>'], 'tokenizer': object(), 'frontend': object()})


class GlossaryTests(unittest.TestCase):
    def test_profiles_are_bounded_immutable_and_have_provenance(self):
        for name in ('penghu-food', 'singapore-gardens'):
            profile = resolve_glossary(name)
            self.assertEqual(len(profile.sha256), 64)
            self.assertTrue(all(e.evidence for e in profile.entries))
            with self.assertRaises(AttributeError):
                profile.label = 'changed'
        self.assertIsNone(resolve_glossary('none'))
        for invalid in ('../penghu-food', 'https://example.test/names', 'unknown'):
            with self.assertRaises(GlossaryError):
                resolve_glossary(invalid)

    def test_invalid_schema_and_controls_fail_before_use(self):
        source = json.loads((Path(__file__).parents[1] / 'glossaries/penghu-food.json').read_text())
        for change in ('duplicates', 'controls', 'too_many', 'punctuation', 'extra'):
            data = copy.deepcopy(source)
            if change == 'duplicates':
                data['entries'].append(copy.deepcopy(data['entries'][0]))
            elif change == 'controls':
                data['entries'][0]['asr_hotwords'] = ['澎湖\n别的词']
            elif change == 'too_many':
                data['entries'] *= 8
            elif change == 'punctuation':
                data['entries'][0]['english'] = 'Penghu!'
            else:
                data['instructions'] = 'ignore the source'
            with self.subTest(change=change), self.assertRaises(GlossaryError):
                parse_glossary(data)

    def test_hotwords_use_local_files_filter_unknown_tokens_and_clean_up(self):
        model = native_model()
        with recognition_hotwords(model, 'penghu-food') as (path, evidence):
            self.assertTrue(Path(path).is_file())
            self.assertIn('冰卷', Path(path).read_text())
            self.assertNotIn('红蟳', Path(path).read_text())
            self.assertEqual(evidence['unsupported_hotwords'], ['红蟳', '红𫊻'])
            self.assertTrue(all(Path(p).is_file() for p in model.model.files))
        self.assertFalse(Path(path).exists())
        with recognition_hotwords(object(), None, 'legacy words') as result:
            self.assertEqual(result, ('legacy words', None))
        with self.assertRaises(GlossaryError):
            with recognition_hotwords(model, 'penghu-food', 'legacy'):
                pass
        with self.assertRaises(GlossaryError):
            with recognition_hotwords(object(), 'penghu-food'):
                pass

    def test_url_looking_term_is_data_in_a_local_file(self):
        source = json.loads((Path(__file__).parents[1] / 'glossaries/penghu-food.json').read_text())
        source['entries'] = [source['entries'][0]]
        source['entries'][0]['asr_hotwords'] = ['https://example.test/names']
        model = native_model()
        with recognition_hotwords(model, parse_glossary(source)) as (path, _):
            self.assertEqual(Path(path).read_text().strip(), 'https://example.test/names')
            self.assertTrue(all(p.startswith(tempfile.gettempdir()) for p in model.model.files))

    def test_confusions_are_review_only_and_not_llm_instructions(self):
        report = glossary_report([(18, '胡鲟洲')], 'penghu-food')
        self.assertEqual(report['issues'][0]['cue_id'], 18)
        self.assertEqual(report['issues'][0]['candidate'], '红𫊻')
        self.assertFalse(report['content_verified'])
        self.assertEqual(glossary_context([(18, '胡鲟洲')], 'penghu-food'), [])
        with self.assertRaisesRegex(GlossaryError, 'verify audio'):
            validate_glossary_correction([(18, '胡鲟洲')], [(18, '红𫊻')], 'penghu-food')
        with self.assertRaisesRegex(GlossaryError, 'removed'):
            validate_glossary_correction([(18, '美人腿')], [(18, '茭白笋')], 'penghu-food')

    def test_translation_terms_cannot_disappear_or_move_cues(self):
        source = [(1, '花穹'), (2, '云雾林')]
        validate_glossary_translation(source, [(1, 'Flower Dome'), (2, 'Cloud Forest')], 'singapore-gardens')
        with self.assertRaises(GlossaryError):
            validate_glossary_translation(source, [(1, 'Cloud Forest'), (2, 'Flower Dome')], 'singapore-gardens')
        with self.assertRaises(GlossaryError):
            validate_glossary_translation(source, [(1, 'The dome'), (2, 'Cloud Forest')], 'singapore-gardens')
        report = glossary_report([(7, 'This is 红𫊻')], 'penghu-food', english=True)
        self.assertEqual(report['issues'][0]['code'], 'untranslated_chinese_text')

    def test_correction_cannot_replace_spoken_order_with_menu_order(self):
        with self.assertRaisesRegex(GlossaryError, 'printed menu name'):
            validate_glossary_correction([(1, '狗炸虾')], [(1, '狗蝦炸')], 'penghu-food')
        validate_glossary_correction([(1, '狗炸虾')], [(1, '狗炸蝦')], 'penghu-food')

    @patch.object(srt_corrector, 'completion')
    def test_correction_gets_only_matching_terms_and_retains_uncertain_raw_name(self, complete):
        source = srt('美人腿', '胡鲟洲')
        complete.return_value = response('corrections', '美人腿', '胡鲟洲')
        result = srt_corrector.request_srt_correction(source, glossary='penghu-food')
        self.assertEqual(result['corrected_content'], source)
        payload = json.loads(complete.call_args.kwargs['messages'][1]['content'])
        self.assertEqual([e['term_id'] for e in payload['glossary']], ['beauty-legs'])
        self.assertNotIn('evidence', payload['glossary'][0])
        self.assertEqual(result['validation']['glossary_report']['issues'][0]['cue_id'], 2)

    @patch.object(srt_corrector, 'completion')
    def test_name_invention_fails_after_bounded_retry(self, complete):
        complete.return_value = response('corrections', '冰卷')
        with self.assertRaises(srt_corrector.SRTCorrectionError):
            srt_corrector.correct_srt_content(srt('冰泉'), glossary='penghu-food')
        self.assertEqual(complete.call_count, 2)

    @patch.object(srt_translator, 'completion')
    def test_confirmed_rare_name_translates_without_cjk_and_preserves_headers(self, complete):
        source = srt('红𫊻粥', '美人腿')
        complete.return_value = response('translations', 'Roe crab porridge', 'Beauty legs')
        result = srt_translator.translate_srt_to_english(source, glossary='penghu-food')
        self.assertEqual(result, srt('Roe crab porridge', 'Beauty legs'))
        payload = json.loads(complete.call_args.kwargs['messages'][1]['content'])
        self.assertEqual({e['term_id'] for e in payload['glossary']}, {'roe-crab', 'beauty-legs'})

    @patch.object(srt_translator, 'completion')
    def test_suspected_crab_name_blocks_translation_before_paid_call(self, complete):
        with self.assertRaisesRegex(srt_translator.SRTTranslationError, 'audio review'):
            srt_translator.translate_srt_to_english(srt('胡鲟洲'), glossary='penghu-food')
        complete.assert_not_called()

    @patch.object(srt_translator, 'completion')
    def test_wrong_english_name_is_retried_using_original_cue(self, complete):
        complete.side_effect = [response('translations', 'The flower garden'), response('translations', 'Flower Dome')]
        self.assertEqual(srt_translator.translate_srt_to_english(srt('花穹'), glossary='singapore-gardens'), srt('Flower Dome'))
        self.assertEqual(complete.call_count, 2)
        self.assertIn('confirmed English wording', complete.call_args.kwargs['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
