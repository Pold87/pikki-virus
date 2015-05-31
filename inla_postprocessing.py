import pandas as pd
import numpy as np


submission = pd.read_csv("sampleSubmission.csv")
inla_sub = pd.read_csv("output_inla.csv")


values = inla_sub.iloc[:, 1]
values = values + 0.25

submission.WnvPresent = values

submission.to_csv("inla_postprocessed.csv", index=False)
