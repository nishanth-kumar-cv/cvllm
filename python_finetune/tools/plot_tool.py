from langchain.tools import Tool
import pandas as pd
import plotly.express as px
import plotly.io as pio
import uuid
import os
from difflib import get_close_matches

PLOT_DIR = "frontend/static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

class PlotTool:
    name = "Plot Generator"
    description = "Use this to generate interactive plots from user CSV files"

    def fuzzy_column_match(self, keyword, columns):
        return get_close_matches(keyword.lower(), [c.lower() for c in columns], n=1, cutoff=0.5)[0] if get_close_matches(keyword.lower(), columns, n=1) else columns[0]

    def run(self, query: str) -> str:
        csv_path = "../data/business_docs/sample_data.csv"
        df = pd.read_csv(csv_path)
        x_col = self.fuzzy_column_match("month", df.columns)
        y_col = self.fuzzy_column_match("sales", df.columns)

        if "bar" in query.lower():
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        elif "line" in query.lower():
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")

        plot_id = str(uuid.uuid4())
        html_path = os.path.join(PLOT_DIR, f"{plot_id}.html")
        png_path = os.path.join(PLOT_DIR, f"{plot_id}.png")
        csv_path_out = os.path.join(PLOT_DIR, f"{plot_id}.csv")

        fig.write_html(html_path)
        pio.write_image(fig, png_path, format='png')
        df[[x_col, y_col]].to_csv(csv_path_out, index=False)

        return f"/static/plots/{plot_id}.html|{plot_id}"

plot_tool = Tool(name="PlotTool", func=PlotTool().run, description=PlotTool.description)
