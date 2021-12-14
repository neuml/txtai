"""
SQL module tests
"""

import unittest

from txtai.database import SQL, SQLException, SQLite


class TestSQL(unittest.TestCase):
    """
    Tests SQL parsing and generation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        # Create SQL parser for SQLite
        cls.db = SQLite({})
        cls.db.initialize()

        cls.sql = SQL(cls.db)

    def testBadSQL(self):
        """
        Test invalid SQL
        """

        with self.assertRaises(SQLException):
            self.db.search("select * from txtai where order by")

        with self.assertRaises(SQLException):
            self.db.search("select * from txtai where groupby order by")

        with self.assertRaises(SQLException):
            self.db.search("select * from txtai where a(1)")

        with self.assertRaises(SQLException):
            self.db.search("select a b c from txtai where id match id")

    def testGroupby(self):
        """
        Test group by clauses
        """

        prefix = "select count(*), flag from txtai "

        self.assertSql("groupby", prefix + "group by text", "text")
        self.assertSql("groupby", prefix + "group by distinct(a)", 'distinct(json_extract(data, "$.a"))')
        self.assertSql("groupby", prefix + "where a > 1 group by text", "text")

    def testHaving(self):
        """
        Test having clauses
        """

        prefix = "select count(*), flag from txtai "

        self.assertSql("having", prefix + "group by text having count(*) > 1", "count(*) > 1")
        self.assertSql("having", prefix + "where flag = 1 group by text having count(*) > 1", "count(*) > 1")

    def testLimit(self):
        """
        Test limit clauses
        """

        prefix = "select count(*) from txtai "

        self.assertSql("limit", prefix + "limit 100", "100")

    def testOrderby(self):
        """
        Test order by clauses
        """

        prefix = "select * from txtai "

        self.assertSql("orderby", prefix + "order by id", "s.id")
        self.assertSql("orderby", prefix + "order by id, text", "s.id, text")
        self.assertSql("orderby", prefix + "order by id asc", "s.id asc")
        self.assertSql("orderby", prefix + "order by id desc", "s.id desc")
        self.assertSql("orderby", prefix + "order by id asc, text desc", "s.id asc, text desc")

    def testSelectBasic(self):
        """
        Test basic select clauses
        """

        self.assertSql("select", "select id, indexid, tags from txtai", "s.id, s.indexid, s.tags")
        self.assertSql("select", "select id, indexid, flag from txtai", 's.id, s.indexid, json_extract(data, "$.flag") as "flag"')
        self.assertSql("select", "select id, indexid, a.b.c from txtai", 's.id, s.indexid, json_extract(data, "$.a.b.c") as "a.b.c"')
        self.assertSql("select", "select 'id', [id], (id) from txtai", "'id', [s.id], (s.id)")
        self.assertSql("select", "select * from txtai", "*")

    def testSelectCompound(self):
        """
        Test compound select clauses
        """

        self.assertSql("select", "select a + 1 from txtai", 'json_extract(data, "$.a") + 1 as "a"')
        self.assertSql("select", "select 1 * a from txtai", '1 * json_extract(data, "$.a") as "a"')
        self.assertSql("select", "select a/1 from txtai", 'json_extract(data, "$.a") / 1 as "a"')
        self.assertSql("select", "select avg(a-b) from txtai", 'avg(json_extract(data, "$.a") - json_extract(data, "$.b"))')
        self.assertSql("select", "select distinct(text) from txtai", "distinct(text)")
        self.assertSql("select", "select id, score, (a/2)*3 from txtai", 's.id, score, (json_extract(data, "$.a") / 2) * 3')

    def testWhereBasic(self):
        """
        Test basic where clauses
        """

        prefix = "select * from txtai "

        self.assertSql("where", prefix + "where a = b", 'json_extract(data, "$.a") = json_extract(data, "$.b")')
        self.assertSql("where", prefix + "where a = b.value", 'json_extract(data, "$.a") = json_extract(data, "$.b.value")')
        self.assertSql("where", prefix + "where a = 1", 'json_extract(data, "$.a") = 1')
        self.assertSql("where", prefix + "WHERE 1 = a", '1 = json_extract(data, "$.a")')
        self.assertSql("where", prefix + "WHERE a LIKE 'abc'", "json_extract(data, \"$.a\") LIKE 'abc'")
        self.assertSql("where", prefix + "WHERE a NOT LIKE 'abc'", "json_extract(data, \"$.a\") NOT LIKE 'abc'")
        self.assertSql("where", prefix + "WHERE a IN (1, 2, 3, b)", 'json_extract(data, "$.a") IN (1, 2, 3, json_extract(data, "$.b"))')
        self.assertSql("where", prefix + "WHERE a is not null", 'json_extract(data, "$.a") is not null')
        self.assertSql("where", prefix + "WHERE score >= 0.15", "score >= 0.15")

    def testWhereCompound(self):
        """
        Test compound where clauses
        """

        prefix = "select * from txtai "

        self.assertSql("where", prefix + "where a > (b + 1)", 'json_extract(data, "$.a") > (json_extract(data, "$.b") + 1)')
        self.assertSql("where", prefix + "where a > func('abc')", "json_extract(data, \"$.a\") > func('abc')")
        self.assertSql(
            "where", prefix + "where (id = 1 or id = 2) and a like 'abc'", "(s.id = 1 or s.id = 2) and json_extract(data, \"$.a\") like 'abc'"
        )
        self.assertSql(
            "where",
            prefix + "where a > f(d(b, c, 1),1)",
            'json_extract(data, "$.a") > f(d(json_extract(data, "$.b"), json_extract(data, "$.c"), 1), 1)',
        )
        self.assertSql("where", prefix + "where (id = 1 AND id = 2) OR indexid = 3", "(s.id = 1 AND s.id = 2) OR s.indexid = 3")
        self.assertSql("where", prefix + "where f(id) = b(id)", "f(s.id) = b(s.id)")
        self.assertSql("where", prefix + "WHERE f(id)", "f(s.id)")

    def testWhereSimilar(self):
        """
        Test similar functions
        """

        prefix = "select * from txtai "

        self.assertSql("where", prefix + "where similar('abc')", "__SIMILAR__0")
        self.assertSql("similar", prefix + "where similar('abc')", [["abc"]])

        self.assertSql("where", prefix + "where similar('abc') AND id = 1", "__SIMILAR__0 AND s.id = 1")
        self.assertSql("similar", prefix + "where similar('abc')", [["abc"]])

        self.assertSql("where", prefix + "where similar('abc') and similar('def')", "__SIMILAR__0 and __SIMILAR__1")
        self.assertSql("similar", prefix + "where similar('abc') and similar('def')", [["abc"], ["def"]])

        self.assertSql("where", prefix + "where similar('abc', 1000)", "__SIMILAR__0")
        self.assertSql("similar", prefix + "where similar('abc', 1000)", [["abc", "1000"]])

        self.assertSql("where", prefix + "where similar('abc', 1000) and similar('def', 10)", "__SIMILAR__0 and __SIMILAR__1")
        self.assertSql("similar", prefix + "where similar('abc', 1000) and similar('def', 10)", [["abc", "1000"], ["def", "10"]])

    def assertSql(self, clause, query, expected):
        """
        Helper method to assert a query clause is as expected

        Args:
            clause: clause to select
            query: input query
            expected: expected transformed query value
        """

        self.assertEqual(self.sql(query)[clause], expected)
