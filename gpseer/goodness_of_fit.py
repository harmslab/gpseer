import numpy as np
import pandas as pd
from epistasis.stats import pearson, split_gpm
from .maximum_likelihood import (
    construct_model,
    fit_ml_model,
    get_ml_predictions_df
)

# class GoodnessOfFit(BaseApp):

#     # ----------------- Goodness of Fit Settings --------------

#     npoints = Int(
#         10,
#         help="List with the number of observations to make at each goodness-of-fit test.",
#         config=True
#     )

#     nsamples = Int(
#         10,
#         help="Number of samples for each test.",
#         config=True
#     )

#     @catch_config_error
#     def start(self):
#         self.zinitialize()

#         # Get data.
#         gpm = self.read_data()

#         start = gpm.length
#         stop = len(gpm.genotypes) - gpm.length - 1
#         if stop <= start:
#             raise Exception("Not enough observations to split the data into training and test sets.")

#         nobs = np.around(
#             np.linspace(start, stop, self.npoints),
#             decimals=0
#         ).astype(int)

#         train_scores = np.empty((len(nobs), self.nsamples), dtype=float)
#         test_scores = np.empty((len(nobs), self.nsamples), dtype=float)

#         for i, n in tqdm(enumerate(nobs)):
#             for j in range(self.nsamples):
#                 train, test = split_gpm(gpm, nobs=n)

#                 self.model.add_gpm(train)
#                 self.model.fit()
#                 out = self.model.predict(X=test.genotypes)

#                 train_scores[i, j] = self.model.score()
#                 test_scores[i, j] = pearson(out, test.phenotypes)**2

#         x = np.array([
#             train_scores.mean(axis=1),
#             test_scores.mean(axis=1)
#         ])

#         output = pd.DataFrame(
#             x.T,
#             index=nobs,
#             columns=[
#                 'training_set_scores',
#                 'test_set_scores'
#             ]
#         )
#         output.to_csv(self.outfile, index_label='nobs')



def run_goodness_of_fit(
        logger,
        input_file,
        output_file,
        wildtype=None,
        threshold=None,
        spline_order=None,
        spline_smoothness=10,
        epistasis_order=1,
        nreplicates=None,
        genotype_file=None,
    ):
    input_df = read_input_file(input_file)
    model = construct_model(
        threshold=threshold,
        spline_order=spline_order,
        spline_smoothness=spline_smoothness,
        epistasis_order=epistasis_order
    )
