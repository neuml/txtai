"""
Token module
"""


class Token:
    """
    Methods to check for token type.
    """

    # Similar token replacement
    SIMILAR_TOKEN = "__SIMILAR__"

    # Default list of comparison operators
    OPERATORS = ["=", "!=", "<>", ">", ">=", "<", "<=", "+", "-", "*", "/", "%", "||", "not", "between", "like", "is", "null"]

    # Default list of logic separators
    LOGIC_SEPARATORS = ["and", "or"]

    # Default alias token
    ALIAS = ["as"]

    # Default list of sort order operators
    SORT_ORDER = ["asc", "desc"]

    @staticmethod
    def get(tokens, x):
        """
        Gets token at position x. This method will validate position is valid within tokens.

        Args:
            tokens: input tokens
            x: position to retrieve

        Returns:
            tokens[x] if x is a valid position, None otherwise
        """

        if 0 <= x < len(tokens):
            return tokens[x]

        return None

    @staticmethod
    def isalias(tokens, x, alias):
        """
        Checks if tokens[x] is an alias keyword.

        Args:
            tokens: input tokens
            x: current position
            alias: if column alias processing is enabled

        Returns:
            True if tokens[x] is an alias token, False otherwise
        """

        prior = Token.get(tokens, x - 1)
        token = tokens[x]

        # True if prior token is not a separator and not a grouping token and current token is either a column token or quoted token
        return alias and x > 0 and not Token.isseparator(prior) and not Token.isgroupstart(prior) and (Token.iscolumn(token) or Token.isquoted(token))

    @staticmethod
    def isattribute(tokens, x):
        """
        Checks if tokens[x] is an attribute.

        Args:
            tokens: input tokens
            x: current position

        Returns:
            True if tokens[x] is an attribute, False otherwise
        """

        # True if token is a column and next token is not an operator
        return Token.iscolumn(tokens[x]) and not Token.isoperator(Token.get(tokens, x + 1))

    @staticmethod
    def isbracket(token):
        """
        Checks if token is an open bracket.

        Args:
            token: token to test

        Returns:
            True if token is an open bracket, False otherwise
        """

        # Token is a bracket
        return token == "["

    @staticmethod
    def iscolumn(token):
        """
        Checks if token is a column name.

        Args:
            token: token to test

        Returns:
            True if this token is a column name token, False otherwise
        """

        # Columns are not operators, logic separators, literals or sort order tokens
        return (
            token
            and not Token.isoperator(token)
            and not Token.islogicseparator(token)
            and not Token.isliteral(token)
            and not Token.issortorder(token)
        )

    @staticmethod
    def iscompound(tokens, x):
        """
        Checks if tokens[x] is a compound expression.

        Args:
            tokens: input tokens
            x: current position

        Returns:
            True if tokens[x] is a compound expression, False otherwise
        """

        # Compound expression is defined as: <column> <operator(s)> <column>
        return Token.isoperator(tokens[x]) and (Token.iscolumn(Token.get(tokens, x - 1)) or Token.iscolumn(Token.get(tokens, x + 1)))

    @staticmethod
    def isfunction(tokens, x):
        """
        Checks if tokens[x] is a function.

        Args:
            tokens: input tokens
            x: current position

        Returns:
            True if tokens[x] is a function, False otherwise
        """

        # True if a column token is followed by an open paren
        return Token.iscolumn(tokens[x]) and Token.get(tokens, x + 1) == "("

    @staticmethod
    def isgroupstart(token):
        """
        Checks if token is a group start token.

        Args:
            token: token to test

        Returns:
            True if token is a group start token, False otherwise
        """

        # Token is a paren
        return token == "("

    @staticmethod
    def isliteral(token):
        """
        Checks if token is a literal.

        Args:
            token: token to test

        Returns:
            True if this token is a literal, False otherwise
        """

        # Literals are wrapped in quotes, parens, wildcards or numeric.
        return token and (token.startswith(("'", '"', ",", "(", ")", "*")) or token.replace(".", "", 1).isdigit())

    @staticmethod
    def islogicseparator(token):
        """
        Checks if token is a logic separator token.

        Args:
            token: token to test

        Returns:
            True if this token is a logic separator, False otherwise
        """

        # Token is a logic separator
        return token and token.lower() in Token.LOGIC_SEPARATORS

    @staticmethod
    def isoperator(token):
        """
        Checks if token is an operator token.

        Args:
            token: token to test

        Returns:
            True if this token is an operator, False otherwise
        """

        # Token is an operator
        return token and token.lower() in Token.OPERATORS

    @staticmethod
    def isquoted(token):
        """
        Checks if token is quoted.

        Args:
            token: token to test

        Returns:
            True if this token is quoted, False otherwise
        """

        # Token is quoted
        return token.startswith(("'", '"')) and token.endswith(("'", '"'))

    @staticmethod
    def isseparator(token):
        """
        Checks if token is a separator token.

        Args:
            token to test

        Returns:
            True if this token is a separator, False otherwise
        """

        # Token is a comma
        return token == ","

    @staticmethod
    def issimilar(tokens, x, similar):
        """
        Checks if tokens[x] is a similar() function.

        Args:
            tokens: input tokens
            x: current position
            similar: list where similar function call parameters are stored, can be None in which case similar processing is skipped

        Returns:
            True if tokens[x] is a similar clause
        """

        # True if a "similar" token is followed by an open paren
        return similar is not None and tokens[x].lower() == "similar" and Token.get(tokens, x + 1) == "("

    @staticmethod
    def issortorder(token):
        """
        Checks if token is a sort order token.

        Args:
            token: token to test

        Returns:
            True if this token is a sort order operator, False otherwise
        """

        # Token is a sort order operator
        return token and token.lower() in Token.SORT_ORDER

    @staticmethod
    def normalize(token):
        """
        Applies a normalization algorithm to the input token as follows:
            - Strip single and double quotes
            - Make lowercase

        Args:
            token: input token

        Returns:
            normalized token
        """

        # Lowercase, replace and return
        return token.lower().replace("'", "").replace('"', "")

    @staticmethod
    def wrapspace(text, token):
        """
        Applies whitespace wrapping rules to token.

        Args:
            text: current text buffer
            token: token to add

        Returns:
            token with whitespace rules applied
        """

        # Wildcards have no whitespace. Need special case since * is also multiply which does have whitespace.
        if token in ["*"] and (not text or text.endswith((" ", "("))):
            return token

        # Operator whitespace
        if Token.isoperator(token) or Token.islogicseparator(token) or token.lower() in ["in"]:
            return f" {token} " if not text.endswith(" ") else f"{token} "

        # Comma whitespace
        if Token.isseparator(token):
            return f"{token} "

        # No whitespace if any of the following is True
        if not text or text.endswith((" ", "(", "[")) or token in ["(", "[", ")", "]"] or token.startswith("."):
            return token

        # Default is to add leading whitespace
        return f" {token}"
