import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("PLL Market Power Ratings")

file = st.file_uploader(label = "Upload the csv file here", type = ["csv"])

def default_instructions():
    st.markdown("We assume each match appears in two rows in the file (so we only use every other row).  Here is how we expect the data to look.  The important columns are 'Team', 'Opponent', 'Spread' (with a positive number meaning 'Team' is favored), and 'Weight.  If the 'Weight' column is missing, all weights are treated as equal.")

    st.markdown("Here is how we expect the file to look.")
    df = pd.read_csv("PLL-template.csv")
    st.write(df)

def process_file(file):
    df = pd.read_csv(file)
    for c in ["Team", "Opponent", "Spread"]:
        if c not in df.columns:
            st.markdown(f"**Error in the uploaded file format**.  The column {c} is missing.")
            default_instructions()
            return None
    
    if "Weight" not in df.columns:
        df["WeightNumber"] = 1
    else:
        df["WeightNumber"] = df["Weight"].str.strip("%").astype(float)/100

    team_coefs = pd.get_dummies(df["Team"], prefix="team", dtype="int")

    # The subtraction doesn't work if some teams are not present
    # For example, "Atlas" is not present in the "Team" column in the template
    for team in df["Opponent"].unique():
        if team not in df["Team"].values:
            team_coefs[f"team_{team}"] = 0

    opp_coefs = -pd.get_dummies(df["Opponent"], prefix="team", dtype="int")

    coefs = team_coefs + opp_coefs

    reg = LinearRegression(fit_intercept=False)
    reg.fit(coefs, df["Spread"], sample_weight=df["WeightNumber"])

    ser = pd.Series(reg.coef_, reg.feature_names_in_)

    st.write(ser)


if file is None:
    default_instructions()
else:
    process_file(file)