import argparse
import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("input", help="The csv file containing the simulation output")
args = parser.parse_args()

df = pd.read_csv(args.input)

print(df)
fig = px.scatter(df, x="t",y="value",color="source",facet_row="name")
fig = px.line(df, x="t",y="value",color="source",facet_row="name")
fig.update_yaxes(matches=None)
fig.show()
