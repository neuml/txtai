"""
Baseball statistics application with txtai and Streamlit.

Install txtai and streamlit (>= 1.23) to run:
  pip install txtai streamlit
"""

import datetime
import math
import os
import random

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from txtai.embeddings import Embeddings


class Stats:
    """
    Base stats class. Contains methods for loading, indexing and searching baseball stats.
    """

    def __init__(self):
        """
        Creates a new Stats instance.
        """

        # Load columns
        self.columns = self.loadcolumns()

        # Load stats data
        self.stats = self.load()

        # Load names
        self.names = self.loadnames()

        # Build index
        self.vectors, self.data, self.embeddings = self.index()

    def loadcolumns(self):
        """
        Returns a list of data columns.

        Returns:
            list of columns
        """

        raise NotImplementedError

    def load(self):
        """
        Loads and returns raw stats.

        Returns:
            stats
        """

        raise NotImplementedError

    def metric(self):
        """
        Primary metric column.

        Returns:
            metric column name
        """

        raise NotImplementedError

    def vector(self, row):
        """
        Build a vector for input row.

        Args:
            row: input row

        Returns:
            row vector
        """

        raise NotImplementedError

    def loadnames(self):
        """
        Loads a name - player id dictionary.

        Returns:
            {player name: player id}
        """

        # Get unique names
        names = {}
        rows = self.stats.sort_values(by=self.metric(), ascending=False)[["nameFirst", "nameLast", "playerID"]].drop_duplicates().reset_index()
        for x, row in rows.iterrows():
            # Name key
            key = f"{row['nameFirst']} {row['nameLast']}"
            key += f" ({row['playerID']})" if key in names else ""

            if key not in names:
                # Scale scores of top n players
                exponent = 2 if ((len(rows) - x) / len(rows)) >= 0.95 else 1

                # score = num seasons ^ exponent
                score = math.pow(len(self.stats[self.stats["playerID"] == row["playerID"]]), exponent)

                # Save name key - values pair
                names[key] = (row["playerID"], score)

        return names

    def index(self):
        """
        Builds an embeddings index to stats data. Returns vectors, input data and embeddings index.

        Returns:
            vectors, data, embeddings
        """

        # Build data dictionary
        vectors = {f'{row["yearID"]}{row["playerID"]}': self.transform(row) for _, row in self.stats.iterrows()}
        data = {f'{row["yearID"]}{row["playerID"]}': dict(row) for _, row in self.stats.iterrows()}

        embeddings = Embeddings(
            {
                "transform": self.transform,
            }
        )

        embeddings.index((uid, vectors[uid], None) for uid in vectors)

        return vectors, data, embeddings

    def metrics(self, name):
        """
        Looks up a player's active years, best statistical year and key metrics.

        Args:
            name: player name

        Returns:
            active, best, metrics
        """

        if name in self.names:
            # Get player stats
            stats = self.stats[self.stats["playerID"] == self.names[name][0]]

            # Build key metrics
            metrics = stats[["yearID", self.metric()]]

            # Get best year, sort by primary metric
            best = int(stats.sort_values(by=self.metric(), ascending=False)["yearID"].iloc[0])

            # Get years active, best year, along with metric trends
            return metrics["yearID"].tolist(), best, metrics

        return range(1871, datetime.datetime.today().year), 1950, None

    def search(self, name=None, year=None, row=None, limit=10):
        """
        Runs an embeddings search. This method takes either a player-year or stats row as input.

        Args:
            name: player name to search
            year: year to search
            row: row of stats to search
            limit: max results to return

        Returns:
            list of results
        """

        if row:
            query = self.vector(row)
        else:
            # Lookup player key and build vector id
            name = self.names.get(name)
            query = f"{year}{name[0] if name else name}"
            query = self.vectors.get(query)

        results, ids = [], set()
        if query is not None:
            for uid, _ in self.embeddings.search(query, limit * 5):
                # Only add unique players
                if uid[4:] not in ids:
                    result = self.data[uid].copy()
                    result["link"] = f'https://www.baseball-reference.com/players/{result["nameLast"].lower()[0]}/{result["bbrefID"]}.shtml'
                    results.append(result)
                    ids.add(uid[4:])

                    if len(ids) >= limit:
                        break

        return results

    def transform(self, row):
        """
        Transforms a stats row into a vector.

        Args:
            row: stats row

        Returns:
            vector
        """

        if isinstance(row, np.ndarray):
            return row

        return np.array([0.0 if not row[x] or np.isnan(row[x]) else row[x] for x in self.columns])


class Batting(Stats):
    """
    Batting stats.
    """

    def loadcolumns(self):
        return [
            "birthMonth",
            "yearID",
            "age",
            "height",
            "weight",
            "G",
            "AB",
            "R",
            "H",
            "1B",
            "2B",
            "3B",
            "HR",
            "RBI",
            "SB",
            "CS",
            "BB",
            "SO",
            "IBB",
            "HBP",
            "SH",
            "SF",
            "GIDP",
            "POS",
            "AVG",
            "OBP",
            "TB",
            "SLG",
            "OPS",
            "OPS+",
        ]

    def load(self):
        # Retrieve raw data from GitHub
        players = pd.read_csv("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/People.csv")
        batting = pd.read_csv("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Batting.csv")
        fielding = pd.read_csv("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Fielding.csv")

        # Merge player data in
        batting = pd.merge(players, batting, how="inner", on=["playerID"])

        # Require player to have at least 350 plate appearances.
        batting = batting[((batting["AB"] + batting["BB"]) >= 350) & (batting["stint"] == 1)]

        # Derive primary player positions
        positions = self.positions(fielding)

        # Calculated columns
        batting["age"] = batting["yearID"] - batting["birthYear"]
        batting["POS"] = batting.apply(lambda row: self.position(positions, row), axis=1)
        batting["AVG"] = batting["H"] / batting["AB"]
        batting["OBP"] = (batting["H"] + batting["BB"]) / (batting["AB"] + batting["BB"])
        batting["1B"] = batting["H"] - batting["2B"] - batting["3B"] - batting["HR"]
        batting["TB"] = batting["1B"] + 2 * batting["2B"] + 3 * batting["3B"] + 4 * batting["HR"]
        batting["SLG"] = batting["TB"] / batting["AB"]
        batting["OPS"] = batting["OBP"] + batting["SLG"]
        batting["OPS+"] = 100 + (batting["OPS"] - batting["OPS"].mean()) * 100

        return batting

    def metric(self):
        return "OPS+"

    def vector(self, row):
        row["TB"] = row["1B"] + 2 * row["2B"] + 3 * row["3B"] + 4 * row["HR"]
        row["AVG"] = row["H"] / row["AB"]
        row["OBP"] = (row["H"] + row["BB"]) / (row["AB"] + row["BB"])
        row["SLG"] = row["TB"] / row["AB"]
        row["OPS"] = row["OBP"] + row["SLG"]
        row["OPS+"] = 100 + (row["OPS"] - self.stats["OPS"].mean()) * 100

        return self.transform(row)

    def positions(self, fielding):
        """
        Derives primary positions for players.

        Args:
            fielding: fielding data

        Returns:
            {player id: (position, number of games)}
        """

        positions = {}
        for _, row in fielding.iterrows():
            uid = f'{row["yearID"]}{row["playerID"]}'
            position = row["POS"] if row["POS"] else 0
            if position == "P":
                position = 1
            elif position == "C":
                position = 2
            elif position == "1B":
                position = 3
            elif position == "2B":
                position = 4
            elif position == "3B":
                position = 5
            elif position == "SS":
                position = 6
            elif position == "OF":
                position = 7

            # Save position if not set or player played more at this position
            if uid not in positions or positions[uid][1] < row["G"]:
                positions[uid] = (position, row["G"])

        return positions

    def position(self, positions, row):
        """
        Looks up primary position for player row.

        Arg:
            positions: all player positions
            row: player row

        Returns:
            primary player positions
        """

        uid = f'{row["yearID"]}{row["playerID"]}'
        return positions[uid][0] if uid in positions else 0


class Pitching(Stats):
    """
    Pitching stats.
    """

    def loadcolumns(self):
        return [
            "birthMonth",
            "yearID",
            "age",
            "height",
            "weight",
            "W",
            "L",
            "G",
            "GS",
            "CG",
            "SHO",
            "SV",
            "IPouts",
            "H",
            "ER",
            "HR",
            "BB",
            "SO",
            "BAOpp",
            "ERA",
            "IBB",
            "WP",
            "HBP",
            "BK",
            "BFP",
            "GF",
            "R",
            "SH",
            "SF",
            "GIDP",
            "WHIP",
            "WADJ",
        ]

    def load(self):
        # Retrieve raw data from GitHub
        players = pd.read_csv("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/People.csv")
        pitching = pd.read_csv("https://raw.githubusercontent.com/chadwickbureau/baseballdatabank/master/core/Pitching.csv")

        # Merge player data in
        pitching = pd.merge(players, pitching, how="inner", on=["playerID"])

        # Require player to have 20 appearances
        pitching = pitching[(pitching["G"] >= 20) & (pitching["stint"] == 1)]

        # Calculated columns
        pitching["age"] = pitching["yearID"] - pitching["birthYear"]
        pitching["WHIP"] = (pitching["BB"] + pitching["H"]) / (pitching["IPouts"] / 3)
        pitching["WADJ"] = (pitching["W"] + pitching["SV"]) / (pitching["ERA"] + pitching["WHIP"])

        return pitching

    def metric(self):
        return "WADJ"

    def vector(self, row):
        row["WHIP"] = (row["BB"] + row["H"]) / (row["IPouts"] / 3) if row["IPouts"] else None
        row["WADJ"] = (row["W"] + row["SV"]) / (row["ERA"] + row["WHIP"]) if row["ERA"] and row["WHIP"] else None

        return self.transform(row)


class Application:
    """
    Main application.
    """

    def __init__(self):
        """
        Creates a new application.
        """

        # Batting stats
        self.batting = Batting()

        # Pitching stats
        self.pitching = Pitching()

    def run(self):
        """
        Runs a Streamlit application.
        """

        st.title("⚾ Baseball Statistics")
        st.markdown(
            """
            This application finds the best matching historical players using vector search with [txtai](https://github.com/neuml/txtai).
            Raw data is from the [Baseball Databank](https://github.com/chadwickbureau/baseballdatabank) GitHub project. Read [this
            article](https://medium.com/neuml/explore-baseball-history-with-vector-search-5778d98d6846) for more details.
        """
        )

        player, search = st.tabs(["Player", "Search"])

        # Player tab
        with player:
            self.player()

        # Search
        with search:
            self.search()

    def player(self):
        """
        Player tab.
        """

        st.markdown("Match by player-season. Each player search defaults to the best season sorted by OPS or Wins Adjusted.")

        # Get parameters
        params = self.params()

        # Category and stats
        category = self.category(params.get("category"), "category")
        stats = self.batting if category == "Batting" else self.pitching

        # Player name
        name = self.name(stats.names, params.get("name"))

        # Player metrics
        active, best, metrics = stats.metrics(name)

        # Player year
        year = self.year(active, params.get("year"), best)

        # Display metrics chart
        if len(active) > 1:
            self.chart(category, metrics)

        # Run search
        results = stats.search(name, year)

        # Display results
        self.table(results, ["link", "nameFirst", "nameLast", "teamID"] + stats.columns[1:])

        # Save parameters
        st.experimental_set_query_params(category=category, name=name, year=year)

    def search(self):
        """
        Stats search tab.
        """

        st.markdown("Find players with similar statistics.")

        stats, category = None, self.category("Batting", "searchcategory")
        with st.form("search"):
            if category == "Batting":
                stats, columns = self.batting, self.batting.columns[:-6]
            elif category == "Pitching":
                stats, columns = self.pitching, self.pitching.columns[:-2]

            # Enter stats with data editor
            inputs = st.data_editor(pd.DataFrame([dict((column, None) for column in columns)]), hide_index=True).astype(float)

            submitted = st.form_submit_button("Search")
            if submitted:
                # Run search
                results = stats.search(row=inputs.to_dict(orient="records")[0])

                # Display table
                self.table(results, ["link", "nameFirst", "nameLast", "teamID"] + stats.columns[1:])

    def params(self):
        """
        Get application parameters. This method combines URL parameters with session parameters.

        Returns:
            parameters
        """

        # Get parameters
        params = st.experimental_get_query_params()
        params = {x: params[x][0] for x in params}

        # Sync parameters with session state
        if all(x in st.session_state for x in ["category", "name", "year"]):
            # Copy session year if category and name are unchanged
            params["year"] = str(st.session_state["year"]) if all(params.get(x) == st.session_state[x] for x in ["category", "name"]) else None

            # Copy category and name from session state
            params["category"] = st.session_state["category"]
            params["name"] = st.session_state["name"]

        return params

    def category(self, category, key):
        """
        Builds category input widget.

        Args:
            category: category parameter
            key: widget key

        Returns:
            category component
        """

        # List of stat categories
        categories = ["Batting", "Pitching"]

        # Get category parameter, default if not available or valid
        default = categories.index(category) if category and category in categories else 0

        # Radio box component
        return st.radio("Stat", categories, index=default, horizontal=True, key=key)

    def name(self, names, name):
        """
        Builds name input widget.

        Args:
            names: list of all allowable names

        Returns:
            name component
        """

        # Get name parameter, default to random weighted value if not valid
        name = name if name and name in names else random.choices(list(names.keys()), weights=[names[x][1] for x in names])[0]

        # Sort names for display
        names = sorted(names)

        # Select box component
        return st.selectbox("Name", names, names.index(name), key="name")

    def year(self, years, year, best):
        """
        Builds year input widget.

        Args:
            years: active years for a player
            year: year parameter
            best: default to best year if year is invalid

        Returns:
            year component
        """

        # Get year parameter, default if not available or valid
        year = int(year) if year and year.isdigit() and int(year) in years else best

        # Slider component
        return int(st.select_slider("Year", years, year, key="year") if len(years) > 1 else years[0])

    def chart(self, category, metrics):
        """
        Displays a metric chart.

        Args:
            category: Batting or Pitching
            metrics: player metrics to plot
        """

        # Key metric
        metric = self.batting.metric() if category == "Batting" else self.pitching.metric()

        # Cast year to string
        metrics["yearID"] = metrics["yearID"].astype(str)

        # Metric over years
        chart = (
            alt.Chart(metrics)
            .mark_line(interpolate="monotone", point=True, strokeWidth=2.5, opacity=0.75)
            .encode(x=alt.X("yearID", title=""), y=alt.Y(metric, scale=alt.Scale(zero=False)))
        )

        # Create metric median rule line
        rule = alt.Chart(metrics).mark_rule(color="gray", strokeDash=[3, 5], opacity=0.5).encode(y=f"median({metric})")

        # Layered chart configuration
        chart = (chart + rule).encode(y=alt.Y(title=metric)).properties(height=200).configure_axis(grid=False)

        # Draw chart
        st.altair_chart(chart + rule, theme="streamlit", use_container_width=True)

    def table(self, results, columns):
        """
        Displays a list of results as a table.

        Args:
            results: list of results
            columns: column names
        """

        if results:
            st.dataframe(
                results,
                column_order=columns,
                column_config={
                    "link": st.column_config.LinkColumn("Link", width="small"),
                    "yearID": st.column_config.NumberColumn("Year", format="%d"),
                    "nameFirst": "First",
                    "nameLast": "Last",
                    "teamID": "Team",
                    "age": "Age",
                    "weight": "Weight",
                    "height": "Height",
                },
            )
        else:
            st.write("Player-Year not found")


@st.cache_resource(show_spinner=False)
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create and run application
    app = create()
    app.run()
