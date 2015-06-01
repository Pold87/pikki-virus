import pandas as pd
import numpy as np


submission = pd.read_csv("sampleSubmission.csv")
inla_sub = pd.read_csv("output_inla_glm.csv")


values = inla_sub.iloc[:, 1]
values = values + 0.4

submission.WnvPresent = values

p# p = pd.read_csv("glm_postprocessed.csv")


submission.to_csv("glm_postprocessed.csv", index=False)
