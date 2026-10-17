import unittest

from funclip.subtitle_meaning import check_meaning_preservation, MeaningPreservationError


class SubtitleMeaningTests(unittest.TestCase):
    def check(self, source, target, mode="correction"):
        return check_meaning_preservation([(1, source)], [(1, target)], mode=mode)

    def test_equivalent_numeric_spelling_and_style_are_allowed(self):
        for source, target in [
            ("高约三十五米", "高约35米"),
            ("低约30%", "低约百分之30"),
            ("下降-3.5%", "下降负百分之3点5"),
            ("價格2,910元", "价格2910元"),
            ("don't change 3.5 percent", "do not change 3 point 5 percent"),
        ]:
            with self.subTest(source=source):self.check(source, target)

    def test_translation_numbers_units_and_natural_wording(self):
        for source, target in [
            ("最后五道菜的价格是2910", "The five dishes cost NT$2,910."),
            ("高约35米", "About thirty-five meters high."),
            ("低约百分之三十", "Around 30 percent lower."),
            ("负3.5%", "negative 3.5 percent"),
            ("一年", "a year"),
            ("一个人", "a person"),
            ("两个人", "both of us"),
            ("第一句", "First line"),
            ("七龙珠", "Dragon Ball Z"),
            ("七美", "Qimei"),
            ("一年四季", "all year round"),
            ("不要再吃了", "Stop eating"),
            ("四五个小时", "Four to five hours"),
            ("两三百个", "two or three hundred"),
            ("两 三个风扇", "two or three fans"),
            ("三分之二", "two-thirds"),
            ("两块半", "two and a half"),
            ("浓郁的味道先出来", "The rich flavor comes first"),
            ("几十年", "decades"),
        ]:
            with self.subTest(source=source):self.check(source, target, "translation")

    def test_changed_amount_sign_percentage_or_units_requires_review(self):
        for source, target in [("2910元", "291元"), ("负3.5%", "3.5%"),
                               ("30%", "30"), ("35米", "35公里"),
                               ("NT$2910", "USD2910"), ("10万", "10"),
                               ("五道菜", "六道菜")]:
            with self.subTest(source=source), self.assertRaises(MeaningPreservationError):
                self.check(source, target)
        with self.assertRaises(MeaningPreservationError):self.check("两块半", "two", "translation")

    def test_negation_added_removed_or_reversed_requires_review(self):
        for source, target in [("那不能进港", "那能进港"), ("那能进港", "那不能进港"),
                               ("不会太冷", "会太冷"), ("没有门票", "有门票"),
                               ("We cannot enter", "We can enter")]:
            with self.subTest(source=source), self.assertRaises(MeaningPreservationError) as error:
                self.check(source, target)
            self.assertIn("negation_changed", [i["code"] for i in error.exception.issues])
        self.check("我们无法进去", "我们不能进去")
        self.check("这是特别的纪念品", "這是特別的紀念品")

    def test_lost_comparison_target_and_approximation_require_review(self):
        for target in ["冷却能耗降低约30%", "比新的建筑低约30%", "比采用传统冷却技术的建筑高约30%"]:
            with self.subTest(target=target), self.assertRaises(MeaningPreservationError):
                self.check("比采用传统冷却技术的建筑低约30%", target)
        with self.assertRaises(MeaningPreservationError):self.check("约30%", "30%")
        with self.assertRaises(MeaningPreservationError):self.check("比传统建筑更低约30%", "比传统建筑更高约30%")

    def test_price_cannot_move_to_another_cue_with_valid_ids(self):
        source = [(300, "谢谢谢谢"), (301, "最后五道菜的价格是2910")]
        moved = [(300, "The final five dishes cost 2910."), (301, "Our first seafood meal in Penghu.")]
        with self.assertRaises(MeaningPreservationError) as error:
            check_meaning_preservation(source, moved, mode="translation")
        self.assertEqual({i["cue_id"] for i in error.exception.issues}, {300, 301})
        correct = [(300, "Thank you"), (301, "The five dishes cost NT$2,910.")]
        report = check_meaning_preservation(source, correct, mode="translation")
        self.assertFalse(report["content_verified"])

    def test_numeric_facts_can_reorder_within_the_same_cue(self):
        self.check("五道菜价格2910", "For 2910 we got five dishes", "translation")

    def test_separated_countdown_digits_remain_individual_facts(self):
        for source, candidate in [
            ("五、四、三、二、一！", "Five four three two one"),
            ("五 四 三 二 一", "5 4 3 2 1"),
            ("三十五、四十", "thirty-five, forty"),
        ]:
            with self.subTest(source=source):
                self.check(source, candidate, "translation")
        for candidate in ["Five four three two", "Five four three two two"]:
            with self.subTest(candidate=candidate), self.assertRaises(MeaningPreservationError):
                self.check("五、四、三、二、一", candidate, "translation")

    def test_comparison_grammar_particle_is_outside_the_target(self):
        self.check("比白天感觉凉爽的很多", "比白天感觉凉爽得很多")
        for candidate in ["比昨天感觉凉爽得很多", "比白天感觉热得很多"]:
            with self.subTest(candidate=candidate), self.assertRaises(MeaningPreservationError):
                self.check("比白天感觉凉爽的很多", candidate)


if __name__ == "__main__":
    unittest.main()
