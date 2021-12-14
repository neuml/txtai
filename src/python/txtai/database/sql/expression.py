"""
Expression module
"""


class Expression:
    """
    Parses expression statements and runs a set of substitution/formatting rules.
    """

    # Similar token replacement
    SIMILAR_TOKEN = "__SIMILAR__"

    # Default list of comparison operators
    OPERATORS = ["=", "!=", "<>", ">", ">=", "<", "<=", "+", "-", "*", "/", "%", "not", "between", "like", "is", "null"]

    # Default list of logic separators
    LOGIC_SEPARATORS = ["and", "or"]

    # Default list of sort order operators
    SORT_ORDER = ["asc", "desc"]

    def __init__(self, resolver):
        """
        Creates a new expression parser.

        Args:
            resolver: function to call to resolve query column names with database column names
        """

        self.resolver = resolver

    def __call__(self, tokens, alias=False, similar=None):
        """
        Parses a list of tokens, replaces query column names with database column names and
        adds similar query placeholders. Returns rewritten clause.

        Args:
            tokens: input expression
            alias: if True, column aliases should be generated
            similar: list of similar queries, if present new similar queries are appended to this list

        Returns:
            rewritten clause as a string
        """

        # Resolve column name tokens
        tokens = self.resolve(list(tokens), alias, similar)

        # Re-write query and return
        return self.build(tokens)

    def resolve(self, tokens, alias, similar):
        """
        Resolves query column names with database column names.

        Args:
            tokens: input expression
            alias: if True, column aliases should be generated
            similar: list of similar queries, if present new similar queries are appended to this list

        Returns:
            resolved tokens
        """

        iterator = enumerate(tokens)
        for x, token in iterator:
            # Check if token is a similar function
            if self.issimilar(tokens, x, similar):
                # Resolve similar expression
                self.similar(iterator, tokens, x, similar)

            # Check if token is a function
            elif self.isfunction(tokens, x):
                # Resolve function expression
                self.function(iterator, tokens, token)

            # Check for attribute column not part of a compound expression
            elif self.iscolumn(token) and not self.isoperator(self.get(tokens, x + 1)):
                # Resolve attribute expression
                self.attribute(tokens, x, alias)

            # Check for compound expressions. Need to resolve left and/or right hand side
            elif self.isoperator(token) and (self.iscolumn(self.get(tokens, x - 1)) or self.iscolumn(self.get(tokens, x + 1))):
                # Resolve compound expression
                self.compound(iterator, tokens, x, alias)

        # Remove replaced tokens
        return [token for token in tokens if token]

    def build(self, tokens):
        """
        Builds a new expression from tokens. This method applies a set of rules to generate whitespace between tokens.

        Args:
            tokens: input expression

        Returns:
            expression text
        """

        # Rebuild expression
        text = ""
        for token in tokens:
            # Write token with whitespace rules applied
            text += self.wrapspace(text, token)

        # Remove any leading/trailing whitespace and return
        return text.strip()

    def issimilar(self, tokens, x, similar):
        """
        Checks if tokens[x] is a similar() function.

        Args:
            tokens: input tokens
            x: current position
            similar: list of similar clauses

        Returns:
            True if tokens[x] is a similar clause
        """

        return similar is not None and tokens[x].lower() == "similar" and self.get(tokens, x + 1) == "("

    def similar(self, iterator, tokens, x, similar):
        """
        Substitutes a similar() function call with a placeholder that can later be used to add
        embeddings query results as a filter.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current position
            similar: list of similar clauses
        """

        # Get function parameters
        params = []

        # Clear token from stream, it will be replaced by a placeholder for the function call
        token = tokens[x]
        tokens[x] = None

        while token and token != ")":
            x, token = next(iterator, (None, None))
            if token and token not in ["(", ",", ")"]:
                # Strip quotes
                params.append(token.replace("'", "").replace('"', ""))

            # Clear token from stream
            tokens[x] = None

        # Add placeholder for embedding similarity results
        tokens[x] = f"{Expression.SIMILAR_TOKEN}{len(similar)}"

        # Save parameters
        similar.append(params)

    def isfunction(self, tokens, x):
        """
        Checks if tokens[x] is a function.

        Args:
            tokens: input tokens
            x: current position

        Returns:
            True if tokens[x] is a function, False otherwise
        """

        # Check if token is a functios
        return not self.isoperator(tokens[x]) and self.get(tokens, x + 1) == "("

    def function(self, iterator, tokens, token):
        """
        Resolves column names within the function's parameters.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            token: current token
        """

        # Consume function parameters
        while token and token != ")":
            x, token = next(iterator, (None, None))
            if self.isfunction(tokens, x):
                # Resolve function parameters that are functions
                self.function(iterator, tokens, token)
            elif self.iscolumn(token):
                # Resolve function parameter
                tokens[x] = self.resolver(tokens[x])

    def attribute(self, tokens, x, alias):
        """
        Resolves an attribute column name.

        Args:
            tokens: input tokens
            x: current token position
            alias: if True, column aliases should be generated
        """

        # Alias name
        name = tokens[x]

        # Resolve attribute expression
        tokens[x] = self.resolver(tokens[x])

        # Add alias
        if alias:
            alias = self.resolver(name, alias=True)
            tokens[x] += alias if alias else ""

    def compound(self, iterator, tokens, x, alias):
        """
        Resolves a compound operator (left side <operator(s)> right side).

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current token position
            alias: if True, column aliases should be generated
        """

        # Alias name
        name = None

        # Resolve left side (left side already had function processing applied through standard loop)
        if self.iscolumn(tokens[x - 1]):
            name = tokens[x - 1]
            tokens[x - 1] = self.resolver(tokens[x - 1])

        # Consume operator(s), handle both single and compound operators, i.e. column NOT LIKE 1
        token = tokens[x]
        while token and self.isoperator(token):
            x, token = next(iterator, (None, None))

        # Resolve right side
        if token and self.iscolumn(token):
            # Need to process functions since it hasn't went through the standard loop yet
            if self.isfunction(tokens, x):
                self.function(iterator, tokens, token)
            else:
                name = token
                tokens[x] = self.resolver(token)

        # Add alias to last token
        if name and alias:
            alias = self.resolver(name, alias=True, compound=True)
            tokens[x] += alias if alias else ""

    def iscolumn(self, token):
        """
        Checks if token is a column name.

        Args:
            token: token to test

        Returns:
            True if this token is a column name token, False otherwise
        """

        return token and not self.isoperator(token) and not self.islogicseparator(token) and not self.isliteral(token) and not self.issortorder(token)

    def isoperator(self, token):
        """
        Checks if token is an operator token.

        Args:
            token: token to test

        Returns:
            True if this token is an operator, False otherwise
        """

        return token and token.lower() in Expression.OPERATORS

    def islogicseparator(self, token):
        """
        Checks if token is a logic separator token.

        Args:
            token: token to test

        Returns:
            True if this token is a logic separator, False otherwise
        """

        return token and token.lower() in Expression.LOGIC_SEPARATORS

    def issortorder(self, token):
        """
        Checks if token is a sort order token.

        Args:
            token: token to test

        Returns:
            True if this token is a sort order operator, False otherwise
        """

        return token and token.lower() in Expression.SORT_ORDER

    def isliteral(self, token):
        """
        Checks if token is a literal. Literals are wrapped in quotes, parens, wildcards or numeric.

        Args:
            token: token to test

        Returns:
            True if this token is a literal, False otherwise
        """

        return token and (token.startswith(("'", '"', ",", "(", ")", "[", "]", "*")) or token.replace(".", "", 1).isdigit())

    def wrapspace(self, text, token):
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
        if self.isoperator(token) or self.islogicseparator(token) or token.lower() in ["in"]:
            return f" {token} " if not text.endswith(" ") else f"{token} "

        # Comma whitespace
        if token in [","]:
            return f"{token} "

        # No whitespace if any of the following is True
        if not text or text.endswith((" ", "(", "[")) or token in ["(", "[", ")", "]"]:
            return token

        # Default is to add leading whitespace
        return f" {token}"

    def get(self, tokens, x):
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
