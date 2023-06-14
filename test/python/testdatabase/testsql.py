"""
SQL module tests
"""

import unittest

from txtai.database import DatabaseFactory, SQL, SQLError


class TestSQL(unittest.TestCase):
    """
    Test SQL parsing and generation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize test data.
        """

        # Create SQL parser for SQLite
        cls.db = DatabaseFactory.create({"content": True})
        cls.db.initialize()

        cls.sql = SQL(cls.db)

    def testAlias(self):
        """
        Test alias clauses
        """

        self.assertSql("select", "select a as a1 from txtai", "json_extract(data, '$.a') as a1")
        self.assertSql("select", "select a 'a1' from txtai", "json_extract(data, '$.a') 'a1'")
        self.assertSql("select", 'select a "a1" from txtai', "json_extract(data, '$.a') \"a1\"")
        self.assertSql("select", "select a a1 from txtai", "json_extract(data, '$.a') a1")
        self.assertSql(
            "select",
            "select a, b as b1, c, d + 1 as 'd1' from txtai",
            "json_extract(data, '$.a') as \"a\", json_extract(data, '$.b') as b1, "
            + "json_extract(data, '$.c') as \"c\", json_extract(data, '$.d') + 1 as 'd1'",
        )
        self.assertSql("select", "select id as myid from txtai", "s.id as myid")
        self.assertSql("select", "select length(a) t from txtai", "length(json_extract(data, '$.a')) t")

        self.assertSql("where", "select id as myid from txtai where myid != 3 and a != 1", "myid != 3 and json_extract(data, '$.a') != 1")
        self.assertSql("where", "select txt T from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")
        self.assertSql("where", "select txt 'T' from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")
        self.assertSql("where", "select txt \"T\" from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")
        self.assertSql("where", "select txt as T from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")
        self.assertSql("where", "select txt as 'T' from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")
        self.assertSql("where", "select txt as \"T\" from txtai where t LIKE '%abc%'", "t LIKE '%abc%'")

        self.assertSql("groupby", "select id as myid, count(*) from txtai group by myid, a", "myid, json_extract(data, '$.a')")
        self.assertSql("orderby", "select id as myid from txtai order by myid, a", "myid, json_extract(data, '$.a')")

    def testBadSQL(self):
        """
        Test invalid SQL
        """

        with self.assertRaises(SQLError):
            self.db.search("select * from txtai where order by")

        with self.assertRaises(SQLError):
            self.db.search("select * from txtai where groupby order by")

        with self.assertRaises(SQLError):
            self.db.search("select * from txtai where a(1)")

        with self.assertRaises(SQLError):
            self.db.search("select a b c from txtai where id match id")

    def testBracket(self):
        """
        Test bracket expressions
        """

        self.assertSql("select", "select [a] from txtai", "json_extract(data, '$.a') as \"a\"")
        self.assertSql("select", "select [a] ab from txtai", "json_extract(data, '$.a') ab")
        self.assertSql("select", "select [abc] from txtai", "json_extract(data, '$.abc') as \"abc\"")
        self.assertSql("select", "select [id], text, score from txtai", "s.id, text, score")
        self.assertSql("select", "select [ab cd], text, score from txtai", "json_extract(data, '$.ab cd') as \"ab cd\", text, score")
        self.assertSql("select", "select [a[0]] from txtai", "json_extract(data, '$.a[0]') as \"a[0]\"")
        self.assertSql("select", "select [a[0].ab] from txtai", "json_extract(data, '$.a[0].ab') as \"a[0].ab\"")
        self.assertSql("select", "select [a[0].c[0]] from txtai", "json_extract(data, '$.a[0].c[0]') as \"a[0].c[0]\"")
        self.assertSql("select", "select avg([a]) from txtai", "avg(json_extract(data, '$.a')) as \"avg([a])\"")

        self.assertSql("where", "select * from txtai where [a b] < 1 or a > 1", "json_extract(data, '$.a b') < 1 or json_extract(data, '$.a') > 1")
        self.assertSql("where", "select [a[0].c[0]] a from txtai where a < 1", "a < 1")
        self.assertSql("groupby", "select * from txtai group by [a]", "json_extract(data, '$.a')")
        self.assertSql("orderby", "select * from txtai where order by [a]", "json_extract(data, '$.a')")

    def testDistinct(self):
        """
        Test distinct expressions
        """

        # Attributes
        self.assertSql("select", "select distinct id from txtai", "distinct s.id")
        self.assertSql("select", "select distinct id as myid from txtai", "distinct s.id as myid")
        self.assertSql("select", "select distinct a from txtai", "distinct json_extract(data, '$.a') as \"a\"")
        self.assertSql("select", "select distinct a.b from txtai", "distinct json_extract(data, '$.a.b') as \"a.b\"")

        # Bracket expression
        self.assertSql("select", "select distinct [ab cd] from txtai", "distinct json_extract(data, '$.ab cd') as \"distinct[ab cd]\"")

        # Function expression
        self.assertSql("select", "select distinct(id) from txtai", 'distinct(s.id) as "distinct(id)"')
        self.assertSql("select", "select count(distinct id) from txtai", 'count(distinct s.id) as "count(distinct id)"')
        self.assertSql("select", "select count(distinct a) from txtai", "count(distinct json_extract(data, '$.a')) as \"count(distinct a)\"")
        self.assertSql("select", "select count(distinct avg(id)) from txtai", 'count(distinct avg(s.id)) as "count(distinct avg(id))"')
        self.assertSql(
            "select", "select count(distinct avg(a)) from txtai", "count(distinct avg(json_extract(data, '$.a'))) as \"count(distinct avg(a))\""
        )

        # Compound expression
        self.assertSql("select", "select distinct a/1 from txtai", "distinct json_extract(data, '$.a') / 1 as \"a / 1\"")
        self.assertSql("select", "select distinct(a/1) from txtai", "distinct(json_extract(data, '$.a') / 1) as \"distinct(a / 1)\"")

    def testGroupby(self):
        """
        Test group by clauses
        """

        prefix = "select count(*), flag from txtai "

        self.assertSql("groupby", prefix + "group by text", "text")
        self.assertSql("groupby", prefix + "group by distinct(a)", "distinct(json_extract(data, '$.a'))")
        self.assertSql("groupby", prefix + "where a > 1 group by text", "text")

    def testHaving(self):
        """
        Test having clauses
        """

        prefix = "select count(*), flag from txtai "

        self.assertSql("having", prefix + "group by text having count(*) > 1", "count(*) > 1")
        self.assertSql("having", prefix + "where flag = 1 group by text having count(*) > 1", "count(*) > 1")

    def testIsSQL(self):
        """
        Test SQL detection method.
        """

        self.assertTrue(self.sql.issql("select text from txtai where id = 1"))
        self.assertFalse(self.sql.issql(1234))

    def testLimit(self):
        """
        Test limit clauses
        """

        prefix = "select count(*) from txtai "

        self.assertSql("limit", prefix + "limit 100", "100")

    def testOffset(self):
        """
        Test offset clauses
        """

        prefix = "select count(*) from txtai "

        self.assertSql("offset", prefix + "limit 100 offset 50", "50")
        self.assertSql("offset", prefix + "offset 50", "50")

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
        self.assertSql("select", "select id, indexid, flag from txtai", "s.id, s.indexid, json_extract(data, '$.flag') as \"flag\"")
        self.assertSql("select", "select id, indexid, a.b.c from txtai", "s.id, s.indexid, json_extract(data, '$.a.b.c') as \"a.b.c\"")
        self.assertSql("select", "select 'id', [id], (id) from txtai", "'id', s.id, (s.id)")
        self.assertSql("select", "select * from txtai", "*")

    def testSelectCompound(self):
        """
        Test compound select clauses
        """

        self.assertSql("select", "select a + 1 from txtai", "json_extract(data, '$.a') + 1 as \"a + 1\"")
        self.assertSql("select", "select 1 * a from txtai", "1 * json_extract(data, '$.a') as \"1 * a\"")
        self.assertSql("select", "select a/1 from txtai", "json_extract(data, '$.a') / 1 as \"a / 1\"")
        self.assertSql("select", "select avg(a-b) from txtai", "avg(json_extract(data, '$.a') - json_extract(data, '$.b')) as \"avg(a - b)\"")
        self.assertSql("select", "select distinct(text) from txtai", "distinct(text)")
        self.assertSql("select", "select id, score, (a/2)*3 from txtai", "s.id, score, (json_extract(data, '$.a') / 2) * 3 as \"(a / 2) * 3\"")
        self.assertSql("select", "select id, score, (a/2*3) from txtai", "s.id, score, (json_extract(data, '$.a') / 2 * 3) as \"(a / 2 * 3)\"")
        self.assertSql(
            "select",
            "select func(func2(indexid + 1), a) from txtai",
            "func(func2(s.indexid + 1), json_extract(data, '$.a')) as \"func(func2(indexid + 1), a)\"",
        )
        self.assertSql("select", "select func(func2(indexid + 1), a) a from txtai", "func(func2(s.indexid + 1), json_extract(data, '$.a')) a")
        self.assertSql("select", "select 'prefix' || id from txtai", "'prefix' || s.id as \"'prefix' || id\"")
        self.assertSql("select", "select 'prefix' || id id from txtai", "'prefix' || s.id id")
        self.assertSql("select", "select 'prefix' || a a from txtai", "'prefix' || json_extract(data, '$.a') a")

    def testSimilar(self):
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

        self.assertSql("where", prefix + "where coalesce(similar('abc'), similar('abc'))", "coalesce(__SIMILAR__0, __SIMILAR__1)")
        self.assertSql("similar", prefix + "where coalesce(similar('abc'), similar('abc'))", [["abc"], ["abc"]])

    def testUpper(self):
        """
        Test SQL statements are case insensitive.
        """

        self.assertSql("groupby", "SELECT * FROM TXTAI WHERE a = 1 GROUP BY id", "s.id")
        self.assertSql("orderby", "SELECT * FROM TXTAI WHERE a = 1 ORDER BY id", "s.id")

    def testWhereBasic(self):
        """
        Test basic where clauses
        """

        prefix = "select * from txtai "

        self.assertSql("where", prefix + "where a = b", "json_extract(data, '$.a') = json_extract(data, '$.b')")
        self.assertSql("where", prefix + "where abc = def", "json_extract(data, '$.abc') = json_extract(data, '$.def')")
        self.assertSql("where", prefix + "where a = b.value", "json_extract(data, '$.a') = json_extract(data, '$.b.value')")
        self.assertSql("where", prefix + "where a = 1", "json_extract(data, '$.a') = 1")
        self.assertSql("where", prefix + "WHERE 1 = a", "1 = json_extract(data, '$.a')")
        self.assertSql("where", prefix + "WHERE a LIKE 'abc'", "json_extract(data, '$.a') LIKE 'abc'")
        self.assertSql("where", prefix + "WHERE a NOT LIKE 'abc'", "json_extract(data, '$.a') NOT LIKE 'abc'")
        self.assertSql("where", prefix + "WHERE a IN (1, 2, 3, b)", "json_extract(data, '$.a') IN (1, 2, 3, json_extract(data, '$.b'))")
        self.assertSql("where", prefix + "WHERE a is not null", "json_extract(data, '$.a') is not null")
        self.assertSql("where", prefix + "WHERE score >= 0.15", "score >= 0.15")

    def testWhereCompound(self):
        """
        Test compound where clauses
        """

        prefix = "select * from txtai "

        self.assertSql("where", prefix + "where a > (b + 1)", "json_extract(data, '$.a') > (json_extract(data, '$.b') + 1)")
        self.assertSql("where", prefix + "where a > func('abc')", "json_extract(data, '$.a') > func('abc')")
        self.assertSql(
            "where", prefix + "where (id = 1 or id = 2) and a like 'abc'", "(s.id = 1 or s.id = 2) and json_extract(data, '$.a') like 'abc'"
        )
        self.assertSql(
            "where",
            prefix + "where a > f(d(b, c, 1),1)",
            "json_extract(data, '$.a') > f(d(json_extract(data, '$.b'), json_extract(data, '$.c'), 1), 1)",
        )
        self.assertSql("where", prefix + "where (id = 1 AND id = 2) OR indexid = 3", "(s.id = 1 AND s.id = 2) OR s.indexid = 3")
        self.assertSql("where", prefix + "where f(id) = b(id)", "f(s.id) = b(s.id)")
        self.assertSql("where", prefix + "WHERE f(id)", "f(s.id)")

    def assertSql(self, clause, query, expected):
        """
        Helper method to assert a query clause is as expected.

        Args:
            clause: clause to select
            query: input query
            expected: expected transformed query value
        """

        self.assertEqual(self.sql(query)[clause], expected)
