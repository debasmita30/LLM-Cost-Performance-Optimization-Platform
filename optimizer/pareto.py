def compute_pareto(df):

    pareto_points = []

    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if (
                other["accuracy"] >= row["accuracy"] and
                other["cost"] <= row["cost"] and
                (
                    other["accuracy"] > row["accuracy"] or
                    other["cost"] < row["cost"]
                )
            ):
                dominated = True
                break

        if not dominated:
            pareto_points.append(row)

    return df.loc[[r.name for r in pareto_points]]