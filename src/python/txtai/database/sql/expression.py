"""
Expression module
"""

from .token import Token


class Expression:
    """
    Parses expression statements and runs a set of substitution/formatting rules.
    """

    def __init__(self, resolver, tolist):
        """
        Creates a new expression parser.

        Args:
            resolver: function to call to resolve query column names with database column names
            tolist: outputs expression lists if True, text if False
        """

        self.resolver = resolver
        self.tolist = tolist

    def __call__(self, tokens, alias=False, aliases=None, similar=None):
        """
        Parses and formats a list of tokens as follows:
            - Replaces query column names with database column names
            - Adds similar query placeholders and extracts similar function parameters
            - Rewrites expression and returns

        Args:
            tokens: input expression
            alias: if True, column aliases should be generated and added to aliases dict
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
            similar: list of similar queries, if present new similar queries are appended to this list

        Returns:
            rewritten clause
        """

        # Processes token expressions and applies a set of transformation rules
        transformed = self.process(list(tokens), alias, aliases, similar)

        # Re-write alias expression and return
        if alias and not self.tolist:
            return self.buildalias(transformed, tokens, aliases)

        # Re-write input expression and return
        return self.buildlist(transformed) if self.tolist is True else self.buildtext(transformed)

    def process(self, tokens, alias, aliases, similar):
        """
        Replaces query column names with database column names, adds similar query placeholders and
        extracts similar function parameters.

        Args:
            tokens: input expression
            alias: if True, column aliases should be generated and added to aliases dict
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
            similar: list of similar queries, if present new similar queries are appended to this list

        Returns:
            transformed tokens
        """

        # Create clause index and token iterator
        index, iterator = 0, enumerate(tokens)
        for x, token in iterator:
            # Check if separator, increment clause index
            if Token.isseparator(token):
                index += 1

            # Check if token is a square bracket
            elif Token.isbracket(token):
                # Resolve bracket expression
                self.bracket(iterator, tokens, x)

            # Check if token is a similar function
            elif Token.issimilar(tokens, x, similar):
                # Resolve similar expression
                self.similar(iterator, tokens, x, similar)

            # Check if token is a function
            elif Token.isfunction(tokens, x):
                # Resolve function expression
                self.function(iterator, tokens, token, aliases)

            # Check for alias expression
            elif Token.isalias(tokens, x, alias):
                # Process alias expression
                self.alias(iterator, tokens, x, aliases, index)

            # Check for attribute expression
            elif Token.isattribute(tokens, x):
                # Resolve attribute expression
                self.attribute(tokens, x, aliases)

            # Check for compound expression
            elif Token.iscompound(tokens, x):
                # Resolve compound expression
                self.compound(iterator, tokens, x, aliases)

        # Remove replaced tokens
        return [token for token in tokens if token]

    def buildtext(self, tokens):
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
            text += Token.wrapspace(text, token)

        # Remove any leading/trailing whitespace and return
        return text.strip()

    def buildlist(self, tokens):
        """
        Builds a new expression from tokens. This method returns a list of expression components. These components can be joined together
        on commas to form a text expression.

        Args:
            tokens: input expression

        Returns:
            expression list
        """

        parts, current, parens, brackets = [], [], 0, 0

        for token in tokens:
            # Create new part
            if token == "," and not parens and not brackets:
                parts.append(self.buildtext(current))
                current = []
            else:
                # Accumulate tokens
                if token == "(":
                    parens += 1
                elif token == ")":
                    parens -= 1
                elif token == "[":
                    brackets += 1
                elif token == "]":
                    brackets -= 1
                elif Token.issortorder(token):
                    token = f" {token}"
                current.append(token)

        # Add last part
        if current:
            parts.append(self.buildtext(current))

        return parts

    def buildalias(self, transformed, tokens, aliases):
        """
        Builds new alias text expression from transformed and input tokens.

        Args:
            transformed: transformed tokens
            tokens: original input tokens
            aliases: dict of column aliases

        Returns:
            alias text expression
        """

        # Convert tokens to expressions
        transformed = self.buildlist(transformed)
        tokens = self.buildlist(tokens)

        expression = []
        for x, token in enumerate(transformed):
            if x not in aliases.values():
                alias = tokens[x]

                # Strip leading/trailing brackets from alias name that doesn't have operators
                if not any(Token.isoperator(t) for t in alias) and alias[0] in ("[", "(") and alias[-1] in ("]", ")"):
                    alias = alias[1:-1]

                # Resolve alias
                token = self.resolver(token, alias)

            expression.append(token)

        # Build alias text expression
        return ", ".join(expression)

    def bracket(self, iterator, tokens, x):
        """
        Consumes a [bracket] expression.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current position
        """

        # Function parameters
        params = []

        # Clear token from stream
        token = tokens[x]
        tokens[x] = None

        # Bracket counter (current token is an open bracket)
        brackets = 1

        # Read until token is a end bracket
        while token and (token != "]" or brackets > 0):
            x, token = next(iterator, (None, None))

            # Increase/decrease bracket counter
            if token == "[":
                brackets += 1
            elif token == "]":
                brackets -= 1

            # Accumulate tokens
            if token != "]" or brackets > 0:
                params.append(token)

            # Clear token from stream
            tokens[x] = None

        # Set last token to resolved bracket expression
        tokens[x] = self.resolve(self.buildtext(params), None)

    def similar(self, iterator, tokens, x, similar):
        """
        Substitutes a similar() function call with a placeholder that can later be used to add
        embeddings query results as a filter.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current position
            similar: list where similar function call parameters are stored
        """

        # Function parameters
        params = []

        # Clear token from stream
        token = tokens[x]
        tokens[x] = None

        # Read until token is a closing paren
        while token and token != ")":
            x, token = next(iterator, (None, None))
            if token and token not in ["(", ",", ")"]:
                # Strip quotes and accumulate tokens
                params.append(token.replace("'", "").replace('"', ""))

            # Clear token from stream
            tokens[x] = None

        # Add placeholder for embedding similarity results
        tokens[x] = f"{Token.SIMILAR_TOKEN}{len(similar)}"

        # Save parameters
        similar.append(params)

    def function(self, iterator, tokens, token, aliases):
        """
        Resolves column names within the function's parameters.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            token: current token
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
        """

        # Consume function parameters
        while token and token != ")":
            x, token = next(iterator, (None, None))
            if Token.isfunction(tokens, x):
                # Resolve function parameters that are functions
                self.function(iterator, tokens, token, aliases)
            elif Token.isattribute(tokens, x):
                # Resolve attributes
                self.attribute(tokens, x, aliases)
            elif Token.iscompound(tokens, x):
                # Resolve compound expressions
                self.compound(iterator, tokens, x, aliases)

    def alias(self, iterator, tokens, x, aliases, index):
        """
        Reads an alias clause and stores it in aliases.

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current position
            aliases: dict where aliases are stored - stores {alias: clause index}
            index: clause index, used to match aliases with columns
        """

        token = tokens[x]

        # If this is an alias token, get next token
        if token in Token.ALIAS:
            x, token = next(iterator, (None, None))

        # Consume tokens until end of stream or a separator is found. Evaluate next token to prevent consuming here.
        while x + 1 < len(tokens) and not Token.isseparator(Token.get(tokens, x + 1)):
            x, token = next(iterator, (None, None))

        # Add normalized alias and clause index
        aliases[Token.normalize(token)] = index

    def attribute(self, tokens, x, aliases):
        """
        Resolves an attribute column name.

        Args:
            tokens: input tokens
            x: current token position
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
        """

        # Resolve attribute expression
        tokens[x] = self.resolve(tokens[x], aliases)

    def compound(self, iterator, tokens, x, aliases):
        """
        Resolves column names in a compound expression (left side <operator(s)> right side).

        Args:
            iterator: tokens iterator
            tokens: input tokens
            x: current token position
            aliases: dict of generated aliases, if present these tokens should NOT be resolved
        """

        # Resolve left side (left side already had function processing applied through standard loop)
        if Token.iscolumn(tokens[x - 1]):
            tokens[x - 1] = self.resolve(tokens[x - 1], aliases)

        # Consume operator(s), handle both single and compound operators, i.e. column NOT LIKE 1
        token = tokens[x]
        while token and Token.isoperator(token):
            x, token = next(iterator, (None, None))

        # Resolve right side
        if token and Token.iscolumn(token):
            # Need to process functions since it hasn't went through the standard loop yet
            if Token.isfunction(tokens, x):
                self.function(iterator, tokens, token, aliases)
            else:
                tokens[x] = self.resolve(token, aliases)

    def resolve(self, token, aliases):
        """
        Resolves this token's value if it is not an alias.

        Args:
            token: token to resolve
            aliases: dict of generated aliases, if present these tokens should NOT be resolved

        Returns:
            resolved token value
        """

        if aliases and Token.normalize(token) in aliases:
            return token

        return self.resolver(token)
