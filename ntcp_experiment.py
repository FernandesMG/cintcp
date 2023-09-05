import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import metrics
from scipy.optimize import curve_fit
from scipy import stats
from time import *
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
import iminuit
from iminuit import Minuit
from bokeh.models import Label, Whisker, Band, ColumnDataSource, LinearAxis, Range1d
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot, row, column, layout
from bokeh.io import output_file, export_svg
from scipy.stats import gaussian_kde


figure_w2h_prop = 1.618
figure_width = 500  # in pixels, 1 px = 0.264583 mm
figure_height = int(figure_width / figure_w2h_prop)

# subfigure_w2h_prop = 1.0
# subfigure_width = 300
# subfigure_height = int(subfigure_width / subfigure_w2h_prop)
subfigure_w2h_prop = 1.33
subfigure_width = 400
subfigure_height = int(subfigure_width / subfigure_w2h_prop)


class NTCPExperiment:

    def __init__(self, data, model_organ, base_organ, test_organ, parameters, n_iterations=500, fraction=1, 
                 ntcp_type='logistic', maxfev_curvefit=5000, force_offset=False):
        """Creates an NTCP experiment object. This object can be used to perform an experiment where a ground truth
        of survival for a dataset is simulated based on a dosimetric parameter (given by a column of ``data``) of an
        organ (``model_organ``) using an NTCP model with the provided parameters (different parameters for different
        ``ntcp_type``). The same dosimetric feature of two given organs referred to as base organ and test organ
        (``base_organ`` and ``test_organ``, also columns of ``data``) are used to fit the simulated ground truth. The
        fitting results of the dosimetric parameter of the two organs are then compared to each other. If the base organ
        is the same as the model organ then the model fitted to base organ should predict the ground truth better than
        the model fitted to the test organ. The dichotomization associated with generating the ground truth turns to
        uncertainty in the fitting procedure which make the test organ predict the ground truth better than the base
        organ in some % of cases leading us to wrongly conclude in those cases that the test organ is the one with the
        actual underlying relationship with survival. This experiment then evaluates how often and in which conditions
        this occurs.
        
        Parameters:
            data: pandas.DataFrame
                The data to be used in the experiment. This table should have dosimetric data for the model organ, 
                the base organ and the organ, each corresponding to a table column.
            model_organ: str
                The column with dosimetric information for the model organ.
            base_organ: str
                The column with dosimetric information for the base organ.
            test_organ: str
                The column with dosimetric information for the test organ.
            parameters: dict
                A dictionary with the parameter values of the ground truth dose-complication relationship and whether 
                these parameters should be fit or the ground truth values should be fixed in the optimization process.
                True=fixed. Example parameters = {'d_50':(1., True)} where d_50 will have value of 1 and won't be
                fitted.
            n_iterations: int
                The number of times fitting is performed in the experiment.
            fraction: float
                The fraction of patients to be drawn from the dataset.
            ntcp_type: str
                The type of ntcp model to use. Options are 'linear' and 'logistic'. If 'linear', the model will only
                have two parameters, the slope and the offset. If 'logistic', the model will have three
                parameters, slope, d_50 and offset.
            maxfev_curvefit: int
                The maximum number of optimization steps. Optimization is least squares.
            force_offset: bool
                Modifier for the NTCP logistic function. Read `compute_log_reg` doc for more info.
        """
        self.data = data.copy()
        self.model_organ = model_organ
        self.base_organ = base_organ
        self.test_organ = test_organ
        self.n_iterations = n_iterations
        self.ntcp_type = ntcp_type
        self.independent_normalization = None
        self.start_time = None
        self.end_time = None
        self.fraction = fraction
        self.npat = round(self.fraction * self.data.shape[0])
        self.maxfev_curvefit = maxfev_curvefit
        self.calculate_bootstrap = None
        self.n_bootstrap_samples = 100
        self.bootstrap_p_value = 0.95
        self.n_split = 1  # number of cross validation folds
        self.parameters = parameters
        self.text_font = 'Calibri'
        self.title_size = '11pt'
        self.text_size = '10pt'

        # where data and results will be stored
        self.parameters_base_list = []  # where the slopes found by fitting to the base organ will be stored
        self.parameters_test_list = []  # where the slopes found by fitting to the test organ will be stored
        self.parameters_unc_base_list = []  # where the uncertainties of base fitting will be stored
        self.parameters_unc_test_list = []  # where the uncertainties of test fitting will be stored
        self.auc_list_test = []  # where the fitting aucs to the test organ will be stored
        self.auc_list_base = []  # where the fitting aucs to the base organ will be sotred
        self.auc_difference = []  # where the difference between the aucs of the base organ and the test organ will be stored
        self.number_better = []  # where whether the fracion of test organ better than base organ is higher than the bootstrap p-value will be stored
        self.number_worse = []  # where whether the fracion of test organ better than base organ is lower than the bootstrap p-value will be stored
        self.events_list = []
        self.aic_list_test = []
        self.aic_list_base = []
        self.aic_difference = []
        self.bic_list_test = []
        self.bic_list_base = []
        self.bic_difference = []

        # Initialize specified function with the parameters defined by the user for ground-truth NTCP curve.
        if self.ntcp_type == 'logistic':
            assert 'd_50' in parameters.keys() and 'gamma' in parameters.keys() and 'offset' in parameters.keys()
            self.d_50 = parameters['d_50'][0]
            self.gamma = parameters['gamma'][0]
            self.offset = parameters['offset'][0]
            self.fixed_d_50 = parameters['d_50'][0] if parameters['d_50'][1] else None
            self.fixed_gamma = parameters['gamma'][0] if parameters['gamma'][1] else None
            self.fixed_offset = parameters['offset'][0] if parameters['offset'][1] else None
            self.force_offset = force_offset
        elif self.ntcp_type == 'linear':
            raise NotImplementedError

    def ntcp(self, x):
        """Computes the ntcp value of x for the chosen ``ntcp_type``. If linear, slope is the only parameter,
        if logistic, slope, gamma and d_50 are the parameters.
        
        Parameters:
            x: numpy.ndarray
                The dosimetric feature values for each patient.
            *args
                See ``ntcp_logistic`` and ``ntcp_linear`` for more info.
                
        Returns:
            y: numpy.ndarray
                The ntcp values for each patient.
        """
        if self.ntcp_type == 'linear':
            raise NotImplementedError
        elif self.ntcp_type == 'logistic':
            y = compute_log_reg(x, self.d_50, self.gamma, self.offset, self.force_offset)
        else:
            raise AttributeError(f'Unknown ntcp_type option, got {self.ntcp_type}')
        return y

    def _normalise_(self, column, factor=None):
        """Normalize the column of data with name ``column`` such that its average is a factor of d_50. This factor is
        defined by the ratio of the column mean and model_col mean. If model_col is None then the factor is 1 and 
        the column average will be d_50.
        
        Parameters:
            column: str
                The column to be normalized.
            factor: float
                The factor to be multiplied by the data[column] mean.
                
        Return:
            float
                The factor by which the data was multiplied.
        """
        old_mean = np.mean(self.data[column])
        self.data[column] = self.d_50 * self.data[column] / old_mean if factor is None else self.data[column] * factor
        new_mean = np.mean(self.data[column])
        print(f'normalising column of dataframe: {column} from {old_mean:.3f} to {new_mean:.3f}')
        return self.d_50 / old_mean

    def normalise_data(self, independent_normalization=False):
        """Normalizes the dosimetric data of the base organ, the model organ and the test organ such that the average
        of the dosimetric feature of each organ is the d_50 value. If independent_normalization is True, then the
        dosimetric data of each organ is normalized independently of each other. Otherwise, if False, then the
        average of the dosimetric feature of the model organ after normalization will be d_50, and the average for
        the other organs will be a x d_50 where a is equal to Xi_mean/Xj_mean, the ratio of the means of the 
        dosimetric features of the two organs where Xi_mean is the mean of the dosimetric feature of the base organ 
        or the organ and Xj_mean is the mean of the dosimetric feature of the model organ.
        
        Parameters:
            independent_normalization: bool
                If True then normalization will be applied to each dosimetric feature independently. Otherwise all
                will be normalized relative to the model organ.
        """
        self.independent_normalization = independent_normalization
        if self.independent_normalization:
            self._normalise_(self.base_organ)
            self._normalise_(self.test_organ)
            if self.model_organ is not None:
                self._normalise_(self.model_organ)
        else:
            model_organ_factor = self._normalise_(self.model_organ)

            if self.base_organ != self.model_organ:
                _ = self._normalise_(self.base_organ, model_organ_factor)
            else:
                print(f'{self.base_organ} already normalised so not again')

            if self.test_organ != self.model_organ:
                _ = self._normalise_(self.test_organ, model_organ_factor)
            else:
                print(f'{self.test_organ} already normalised so not again')

    def permutation_test_between_clfs(self, y_true, y_pred_1, y_pred_2):
        """Calculates bootstrap uncertainty. returns mean difference between the AUC's, and the fraction of 
        the distribution where 1 is better than 2.
        
        Parameters:
            y_true: numpy.ndarray
                The ntcp ground truth.
            y_pred_1: numpy.ndarray
                Estimate number 1 of the ntcp, must be of same size as y_true.
            y_pred_2: numpy.ndarry
                Estimate number 2 of the ntcp, must be of same size as y_true.
                
        Returns:
            float
                Mean difference between the AUC's.
            fraction_better: float
                The fraction of the distribution where estimate 1 is better than estimate 2.
        """

        aggregated_metric_list = []
        score1lst = []
        score2lst = []

        sample_size = len(y_true)
        boot_size = sample_size
        # boot_size = np.round(0.8 * sample_size).astype(int)

        # convert series to np.array if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        for seed in range(self.n_bootstrap_samples):
            aggregated_index = np.random.randint(sample_size, size=boot_size)

            score1 = metrics.roc_auc_score(y_true[aggregated_index], y_pred_1[aggregated_index])
            score2 = metrics.roc_auc_score(y_true[aggregated_index], y_pred_2[aggregated_index])
            aggregated_metric_list.append(score1 - score2)
            score1lst.append(score1)
            score2lst.append(score2)

        # aggregated list is differences between score1 and score2
        fraction_better = len(list(filter(lambda x: (x >= 0), aggregated_metric_list))) / len(aggregated_metric_list)
        # print(fraction_better)
        return np.mean(aggregated_metric_list), fraction_better

    def execute(self, calculate_bootstrap=True, parallel=False, bounds=[(0.1, 5), (0, 10), (0, 1)]):
        """Executes an experiment with one NTCP curve / set of parameter(s). The following list of steps is repeated
        the specified number of times:
        1. A specified fraction of the dataset is selected from the whole dataset.
        2. The ground truth survival probability is generated from the dosimetric parameter of the model organ. This is
           done using the specified NTCP model.
        3. The ground truth survival probability is dichotomized into alive=0 (or no toxicity) or dead=1 (or toxicity)
           at random.
        4. The dosimetric parameters of the base organ and the test organ are fit to the ground truth generated in 2 and
           the respective AUCs are computed and stored as well as the difference between them.
        5. Using bootstrap, the AUC uncertainty spread of both fitted NTCP models is computed and an estimate of the
           frequency with which the test organ has a higher AUC than the base organ (Type I error) is computed.
           (optional)
        6. The frequency computed in 4 is compared to the predefined p-value.
        After all iterations, the percentage of them for which the difference was statistically significant is printed
        out.

        Parameters:
            calculate_bootstrap: bool
                Whether to calculate the uncertainty of the auc results using bootstrap.
            parallel: bool
                If True, computation will occur in parallel leaving one core free.
            bounds: list of tuple
                The bounds to be used for the parameter fits. Default is [(0.1, 5), (0, 10), (0, 1)].
                Parameter order is d_50, gamma, offset.
        """

        def iterate():
            tmp_res = self.data.copy()
            tmp_res = tmp_res.sample(frac=self.fraction)  # draw with replacement is false by default

            # generate survival for the full cohort based on base_organ and fixed ntcp model with slope self.slope
            if self.model_organ is None:
                model_organ_column_name = self.base_organ
            else:
                model_organ_column_name = self.model_organ

            tmp_res['outcome_prob'] = self.ntcp(tmp_res[model_organ_column_name].values)
            # If a~U(0,1) and x in [0, 1], then x>a in x fraction of the times.
            tmp_res['random1'] = np.random.random((self.npat, 1))
            # Dichotomization of the ntcp probability values.
            tmp_res['survival1'] = np.where(tmp_res['outcome_prob'] > tmp_res['random1'], 1, 0)
            # Saving the percentage of patients that died/had toxicity
            tox_percentage = (tmp_res['survival1'] == 1).sum() / self.npat

            # now fit using the first organ
            ntcp_log_reg = LogReg(self.base_organ, force_offset=self.force_offset)
            ntcp_log_reg.bounds = bounds
            ntcp_log_reg.fit(tmp_res, tmp_res['survival1'],
                             self.fixed_d_50, self.fixed_gamma, self.fixed_offset, max_iter=self.maxfev_curvefit)
            auc_base = ntcp_log_reg.score(tmp_res, tmp_res['survival1'])
            parameters_base, parameters_unc_base = [ntcp_log_reg.parameter_values[ntcp_log_reg.parameters_fitted],
                                                    ntcp_log_reg.parameter_devs[ntcp_log_reg.parameters_fitted]]
            pred_base = ntcp_log_reg.predict(tmp_res)
            aic_base = ntcp_log_reg.aic
            bic_base = ntcp_log_reg.bic

            # now fit the test organ
            ntcp_log_reg = LogReg(self.test_organ, force_offset=self.force_offset)
            ntcp_log_reg.bounds = bounds
            ntcp_log_reg.fit(tmp_res, tmp_res['survival1'],
                             self.fixed_d_50, self.fixed_gamma, self.fixed_offset, max_iter=self.maxfev_curvefit)
            auc_test = ntcp_log_reg.score(tmp_res, tmp_res['survival1'])
            parameters_test, parameters_unc_test = [ntcp_log_reg.parameter_values[ntcp_log_reg.parameters_fitted],
                                                    ntcp_log_reg.parameter_devs[ntcp_log_reg.parameters_fitted]]
            pred_test = ntcp_log_reg.predict(tmp_res)
            aic_test = ntcp_log_reg.aic
            bic_test = ntcp_log_reg.bic

            self.calculate_bootstrap = calculate_bootstrap
            if calculate_bootstrap:
                a, fraction_better = self.permutation_test_between_clfs(tmp_res['survival1'], pred_base, pred_test)
            else:
                fraction_better = None

            number_better = 1 if fraction_better > self.bootstrap_p_value else 0
            number_worse = 1 if fraction_better < 1 - self.bootstrap_p_value else 0

            return (tox_percentage, parameters_base, parameters_unc_base, auc_base, parameters_test,
                    parameters_unc_test, auc_test, number_better, number_worse, aic_base, bic_base, aic_test, bic_test)

        self.start_time = time()

        print(f'running with parameters: {[par for par in self.parameters if not self.parameters[par][1]]}, '
              f'bounds: '
              f'{[bound for i, bound in enumerate(bounds) if not self.parameters[list(self.parameters.keys())[i]][1]]}')

        if parallel:
            n_cores = multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1
            results = Parallel(n_jobs=n_cores)(delayed(iterate)() for _ in range(self.n_iterations))
            self.events_list = [result[0] for result in results]
            self.parameters_base_list = [result[1] for result in results]
            self.parameters_unc_base_list = [result[2] for result in results]
            self.auc_list_base = [result[3] for result in results]
            self.parameters_test_list = [result[4] for result in results]
            self.parameters_unc_test_list = [result[5] for result in results]
            self.auc_list_test = [result[6] for result in results]
            self.number_better = [result[7] for result in results]
            self.number_worse = [result[8] for result in results]
            self.auc_difference = [result[3] - result[6] for result in results]
            self.aic_list_base = [result[9] for result in results]
            self.bic_list_base = [result[10] for result in results]
            self.aic_list_test = [result[11] for result in results]
            self.bic_list_test = [result[12] for result in results]
            self.aic_difference = [result[9] - result[11] for result in results]
            self.bic_difference = [result[10] - result[12] for result in results]
        else:
            for i in range(0, self.n_iterations):
                results = iterate()
                self.events_list.append(results[0])
                self.parameters_base_list.append(results[1])
                self.parameters_unc_base_list.append(results[2])
                self.auc_list_base.append(results[3])
                self.parameters_test_list.append(results[4])
                self.parameters_unc_test_list.append(results[5])
                self.auc_list_test.append(results[6])
                self.number_better.append(results[7])
                self.number_worse.append(results[8])
                self.auc_difference.append(results[3] - results[6])
                self.aic_list_base.append(results[9])
                self.bic_list_base.append(results[10])
                self.aic_list_test.append(results[11])
                self.bic_list_test.append(results[12])
                self.aic_difference.append(results[9] - results[11])
                self.bic_difference.append(results[10] - results[12])

        print(f'Percentage of cohorts where {self.base_organ} was significantly (p={self.bootstrap_p_value}) '
              f'better than {self.test_organ}: {np.mean(self.number_better) * 100}')
        print(f'Percentage of cohorts where {self.test_organ} was significantly better (p={self.bootstrap_p_value:.2f})'
              f' than {self.base_organ}: {np.mean(self.number_worse) * 100}')

        self.end_time = time()

    def return_parameters(self):
        """Returns the experiment parameters and the main results, this should be called after execute method has been
        called.

        Returns:
            d: dict
                A dictionary with the values of the parameters and results.
        """

        d = {'slope': self.gamma,
             'base_organ': self.base_organ,
             'organ': self.test_organ,
             'model_organ': self.model_organ,
             'fraction_of_data': self.fraction,
             'mean_auc_base': np.mean(self.auc_list_base),
             'mean_auc_test': np.mean(self.auc_list_test),
             'pearson_correlation': stats.pearsonr(self.data[self.test_organ], self.data[self.base_organ])[0],
             'n_pat': self.npat,
             'zero_offset': self.offset,
             'time': self.end_time - self.start_time,
             'cross_validation_splits': self.n_split,
             'fraction_better': np.mean(self.number_better),
             'fraction_worse': np.mean(self.number_worse),
             'auc_above_zero': len(list(filter(lambda x: (x >= 0), self.auc_difference))) / len(self.auc_difference),
             'n_iterations': self.n_iterations,
             'n_bootstrap_samples': self.n_bootstrap_samples,
             'mean_events_fraction': np.mean(self.events_list),
             'std_events_fraction': np.std(self.events_list, ddof=1),
             'mean_pairwise_auc_difference': np.mean(self.auc_difference),
             'std_pairwise_auc_difference': np.std(self.auc_difference, ddof=1),
             'mean_aic_base': np.mean(self.aic_list_base),
             'std_aic_base': np.std(self.aic_list_base, ddof=1),
             'mean_aic_test': np.mean(self.aic_list_test),
             'std_aic_test': np.std(self.aic_list_test, ddof=1),
             'mean_bic_base': np.mean(self.bic_list_base),
             'std_bic_base': np.std(self.bic_list_base, ddof=1),
             'mean_bic_test': np.mean(self.bic_list_test),
             'std_bic_test': np.std(self.bic_list_test, ddof=1),
             'mean_pairwise_aic_difference': np.mean(self.aic_difference),
             'std_pairwise_aic_difference': np.std(self.aic_difference, ddof=1)
             }

        i = 0
        for par in self.parameters:
            if not self.parameters[par][1]:
                d[f'mean_{par}_base'] = np.mean([item[i] for item in self.parameters_base_list])
                d[f'std_{par}_base'] = np.std([item[i] for item in self.parameters_base_list], ddof=1)
                d[f'mean_{par}_test'] = np.mean([item[i] for item in self.parameters_test_list])
                d[f'std_{par}_test'] = np.std([item[i] for item in self.parameters_test_list], ddof=1)
                i += 1

        return pd.Series(d)

    def return_results_table(self):

        results = [[self.events_list[i], self.parameters_base_list[i], self.parameters_unc_base_list[i],
                    self.auc_list_base[i], self.parameters_test_list[i], self.parameters_unc_test_list[i],
                    self.auc_list_test[i], self.number_better[i], self.number_worse[i],
                    self.auc_difference[i]] for i in range(len(self.events_list))]

        columns = ['Toxicity Percentage', 'Parameters Ground Truth', 'Parameters Ground Truth CI (95%)',
                   'AUC Ground Truth', 'Parameters Alternative', 'Parameters Alternative CI (95%)',
                   'AUC Alternative', 'Significantly Better', 'Significantly Worse', 'AUC GT - AUC AT']

        parameters = ''
        for par in self.parameters:
            parameters += f'{par}:{self.parameters[par][0]}. '
        parameters += f'GT:{self.model_organ}. AT:{self.test_organ}'

        multi_index = pd.MultiIndex.from_arrays(([parameters]*len(columns), columns))

        results_table = pd.DataFrame(results, columns=multi_index, dtype=object)

        return results_table

    def plot(self, plot_difference_correlation=True, save=False, colors=['navy', 'red']):
        """Plots the results of the experiment. This should be called after execute method has been called.
        Red = test organ 
        Blue = reference/base organ.
        
        Parameters:
            plot_difference_correlation: bool
                If true then dosimetric data of test organ will be plotted against the difference between the dosimetric
                data of the base and the test organ.
            save: False or str or pathlib.Path
                If not false, it should be the name of the file to be saved without extension. The format is svg.
            colors: list of str
                The colors to be used in potting. Two are needed. The first one will be attributed to the ground truth,
                the second one to the alternative.
        """

        # correlation plot
        corr_plot = self._get_correlation_plot_(plot_difference_correlation, self.text_font, self.title_size,
                                                self.text_size, colors[0])

        # histograms
        param_hist_plot = self._get_double_hist_(self.data[self.base_organ], self.data[self.test_organ],
                                                 'Dose Parameter Distribution', 'Normalized Dose Parameter', 'Count',
                                                 f'GT: {self.base_organ}', f'AT: {self.test_organ}', 30, self.text_font,
                                                 self.title_size, self.text_size, colors)
        param_hist_plot.x_range.start = 0
        param_hist_plot.y_range.start = 0

        # ntcp model
        ntcp_plot = self._get_ntcp_curves_([0.0, 2.5], self.text_font, self.title_size, self.text_size, colors)

        # histogram of the auc's
        auc_hist_plot = self._get_double_hist_(self.auc_list_base, self.auc_list_test, 'MC Simulation AUC', 'AUC ',
                                               'Count',
                                               f'GT derived model (mean AUC: {np.mean(self.auc_list_base):.3f})',
                                               f'AT derived model (mean AUC: {np.mean(self.auc_list_test):.3f})',
                                               40, self.text_font, self.title_size, self.text_size, colors)
        auc_hist_plot.y_range.start = 0

        auc_diff_hist_plot = self._get_hist_(np.array(self.auc_difference) * 100, 'MC Simulation AUC Difference',
                                             'GT AUC - AT AUC (%)', 'Count', 40, self.text_font, self.title_size,
                                             self.text_size, colors[0])
        auc_diff_hist_plot.y_range.start = 0

        # histogram of the fitted parameters
        par_plots = dict()
        fitted_param_idx = 0
        for i, par in enumerate(self.parameters):
            if not self.parameters[par][1]:  # parameter not fixed i.e. was fitted
                base_par_vals = [item[fitted_param_idx] for item in self.parameters_base_list]
                test_par_vals = [item[fitted_param_idx] for item in self.parameters_test_list]
                par_plots[par] = self._get_double_hist_(
                    base_par_vals, test_par_vals, f'{par} Distribution', par, 'Count',
                    f'GT derived model (mean: {self._get_average_ntcp_(self.parameters_base_list)[i]:.2f})',
                    f'AT derived model (mean: {self._get_average_ntcp_(self.parameters_test_list)[i]:.2f})',
                    40, self.text_font, self.title_size, self.text_size, colors)
                par_plots[par].y_range.start = 0
                fitted_param_idx += 1

        # Additional info as text
        text_plot = self._get_plot_text_()

        plots = [corr_plot, param_hist_plot, ntcp_plot, auc_hist_plot, auc_diff_hist_plot,  *par_plots.values(),
                 text_plot]
        plots = gridplot(plots, ncols=2)

        show(plots)
        if save:
            export_svg(plots, filename=f'{save}.svg')

    def _get_correlation_plot_(self, difference, text_font='Calibri', title_size='11pt', text_size='9pt', color='navy'):
        corr, p_value = stats.pearsonr(self.data[self.test_organ], self.data[self.base_organ])
        corr_fig = figure(
            title=f'GT {self.base_organ} vs AT {self.test_organ} (corr: {corr:.2f})', height=subfigure_height,
            width=subfigure_width, x_axis_label=f"{self.test_organ} normalized",
            y_axis_label=(f'{self.base_organ} - {self.test_organ} both normalized'
                          if difference else f'{self.base_organ} normalized'), output_backend='svg')
        y = (self.data[self.base_organ] - self.data[self.test_organ] if difference else self.data[self.base_organ])
        x = self.data[self.test_organ]
        corr_fig.grid.grid_line_color = "white"
        corr_fig.circle(x, y, size=6, fill_color=color, line_color=color, fill_alpha=0.5, line_alpha=0.2)
        corr_fig = edit_all_plot_text(corr_fig, text_font, title_size, text_size)
        return corr_fig

    @staticmethod
    def _get_hist_(data, title, x_label, y_label, bins=30, text_font='Calibri', title_size='11pt', text_size='9pt',
                   color='navy'):
        data = np.array(data)
        hist, edges = np.histogram(data, bins=bins, range=[data.min(), data.max()])
        x = np.linspace(data.min(), data.max(), 1000)
        kde = gaussian_kde(data)
        y = kde(x) * (edges[1]-edges[0]) * data.size
        hist_fig = figure(plot_height=subfigure_height, plot_width=subfigure_width, title=title, x_axis_label=x_label,
                          y_axis_label=y_label, output_backend='svg')
        hist_fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color=color,
                      line_color=color, fill_alpha=0.5, line_alpha=0.2)
        hist_fig.line(x, y, line_color=color, line_width=1.5, alpha=0.7)
        hist_fig.grid.grid_line_color = "white"
        hist_fig = edit_all_plot_text(hist_fig, text_font, title_size, text_size)
        return hist_fig

    @staticmethod
    def _get_double_hist_(gt_data, at_data, title, x_label, y_label, gt_legend, at_legend, bins=30, text_font='Calibri',
                          title_size='11pt', text_size='9pt', colors=['navy', 'red']):
        gt_data = np.array(gt_data)
        at_data = np.array(at_data)
        x_min = min([gt_data.min(), at_data.min()])
        x_max = max([gt_data.max(), at_data.max()])
        hist_gt, edges_gt = np.histogram(gt_data, bins=bins, range=[x_min, x_max])
        hist_at, edges_at = np.histogram(at_data, bins=bins, range=[x_min, x_max])
        x = np.linspace(x_min, x_max, 1000)
        kde_gt = gaussian_kde(gt_data)
        y_gt = kde_gt(x) * (edges_gt[1]-edges_at[0]) * gt_data.size
        kde_at = gaussian_kde(at_data)
        y_at = kde_at(x) * (edges_gt[1]-edges_at[0]) * gt_data.size
        y_max = max([hist_gt.max(), hist_at.max()])

        hist_fig = figure(plot_height=subfigure_height, plot_width=subfigure_width, title=title, x_axis_label=x_label,
                          y_axis_label=y_label, output_backend='svg')
        hist_fig.quad(top=hist_gt, bottom=0, left=edges_gt[:-1], right=edges_gt[1:], fill_color=colors[0],
                      line_color=colors[0], fill_alpha=0.5, line_alpha=0.2, legend_label=gt_legend)
        hist_fig.quad(top=hist_at, bottom=0, left=edges_at[:-1], right=edges_at[1:], fill_color=colors[1],
                      line_color=colors[1], fill_alpha=0.5, line_alpha=0.2, legend_label=at_legend)
        hist_fig.line(x, y_gt, line_color=colors[0], line_width=1.5, alpha=0.7)
        hist_fig.line(x, y_at, line_color=colors[1], line_width=1.5, alpha=0.7)
        hist_fig.grid.grid_line_color = "white"
        hist_fig.legend.border_line_color = None
        hist_fig.legend.background_fill_alpha = 0.1
        hist_fig.y_range.end = y_max + y_max / 3  # extra space for legend
        hist_fig = edit_all_plot_text(hist_fig, text_font, title_size, text_size)
        return hist_fig

    def _get_ntcp_curves_(self, x_range=[0.0, 2.5], text_font='Calibri', title_size='11pt', text_size='9pt',
                          colors=['navy', 'red']):
        x = np.linspace(x_range[0], x_range[1], 1000)
        y_gt = compute_log_reg(x, *self._get_average_ntcp_(self.parameters_base_list), self.force_offset)
        y_at = compute_log_reg(x, *self._get_average_ntcp_(self.parameters_test_list), self.force_offset)
        y = compute_log_reg(x, *[self.parameters[par][0] for par in self.parameters], self.force_offset)
        initial_parameters_str = ''
        for par in self.parameters:
            initial_parameters_str += f'{par}: {self.parameters[par][0]}. '
        ntcp_fig = figure(plot_height=subfigure_height, plot_width=subfigure_width,
                          title=f'Mean NTCP (GT: {initial_parameters_str})', x_range=x_range, y_range=[0, 1],
                          x_axis_label='Normalized Dose', y_axis_label='NTCP', output_backend='svg')
        ntcp_fig.x_range.start = 0
        ntcp_fig.line(x, y_gt, line_color=colors[0], legend_label=f'Mean GT derived model')
        ntcp_fig.line(x, y_at, line_color=colors[1], legend_label=f'Mean AT derived model')
        ntcp_fig.line(x, y, line_color=colors[0], line_dash='dashed', legend_label=f'Hypothesized NTCP')
        ntcp_fig.legend.location = "top_left"
        ntcp_fig.grid.grid_line_color = "white"
        ntcp_fig.legend.border_line_color = 'white'
        ntcp_fig = edit_all_plot_text(ntcp_fig, text_font, title_size, text_size)
        return ntcp_fig

    def _get_average_ntcp_(self, parameter_list):
        average_parameters = np.mean(parameter_list, axis=0)
        fitted_param_idx = 0
        ntcp_parameters = []
        for i in self.parameters:
            if not self.parameters[i][1]:
                ntcp_parameters.append(average_parameters[fitted_param_idx])
                fitted_param_idx += 1
            else:
                ntcp_parameters.append(self.parameters[i][0])
        return ntcp_parameters

    def _get_plot_text_(self, text_font='Calibri', text_size='10pt'):
        text_plot = figure(width=subfigure_width, height=subfigure_height, x_range=[0, 4], y_range=[0, 4.5],
                           toolbar_location=None, output_backend='svg')
        label_1 = Label(x=0.15, y=3.6, text=f'GT parameter: {self.model_organ}', text_font=text_font,
                        text_font_size=text_size)
        label_2 = Label(x=0.15, y=3.1, text=f'AT parameter: {self.test_organ}', text_font=text_font,
                        text_font_size=text_size)
        label_3 = Label(x=0.15, y=2.6, text=f'Number of iterations: {self.n_iterations}', text_font=text_font,
                        text_font_size=text_size)
        label_4 = Label(x=0.15, y=2.1, text=f'Number of patients: {self.npat}', text_font=text_font,
                        text_font_size=text_size)
        label_5 = Label(x=0.15, y=1.6, text=f'fraction GT significantly better: {np.mean(self.number_better):.3f}',
                        text_font=text_font, text_font_size=text_size)
        label_6 = Label(x=0.15, y=1.1, text=f'fraction AT significantly better: {np.mean(self.number_worse):.3f}',
                        text_font=text_font, text_font_size=text_size)
        label_7 = Label(x=0.15, y=0.6, text=f'fraction no significant difference: '
                                            f'{1-np.mean(self.number_worse)-np.mean(self.number_better):.3f}',
                        text_font=text_font, text_font_size=text_size)
        label_8 = Label(x=0.15, y=0.1, text=f'mean AUC difference: {np.mean(self.auc_difference):.3f} +/-'
                                            f' {np.std(self.auc_difference, ddof=1):.3f}', text_font=text_font,
                        text_font_size=text_size)
        labels = [label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8]
        for label in labels:
            text_plot.add_layout(label)

        text_plot.axis.visible = False
        text_plot.grid.visible = False
        text_plot.outline_line_color = None
        return text_plot


class LogReg(BaseEstimator):
    """LogisticRegression regressor.

    This regressor class inherits from scikit-learn's BaseEstimator. It allows to create regressor objects that fit a
    logistic regression curve to one dimensional data. The Logistic Regression model can be used as an NTCP model with
    2 parameters, plus one added to this implementation:
        -d_50: dose that leads to 50% complication when given to the whole organ. Limits are 0 to infinity.
        -gamma: the slope at d_50.
        -offset (added): the baseline NTCP. Limits are 0 to 1. Even with offset set to zero the model can fit non zero
        intercepts. Use offset only if strictly needed.
    The parameter optimization is done by likelihood maximization.

    Parameters:
        d_label: str
            The label of the covariate to be used in the input pandas dataframe. In the NTCP context this
            covariate would be the equivalent uniform dose or another dosimetric parameter. Default is 'd'.
        verbose: bool
            If True output will be verbose.
        random_state: int
            The seed for the random generator. If None the seed will be random.
        force_offset: bool
            Modifier for the NTCP logistic function. Read `compute_log_reg` doc for more info.
    """
    def __init__(self, d_label='d', verbose=False, random_state=None, force_offset=False):
        self.d_label = d_label
        self.verbose = verbose
        self.random_state = random_state
        self.bounds = [(None, None), (None, None), (0, 1)]
        self.maximum_likelihood_ = None
        self.minimum_negative_log_likelihood_ = None
        self.parameter_order = np.array(['d_50', 'gamma', 'offset'])
        self.parameter_values = np.array([None, None, None])
        self.parameter_devs = np.array([self.d_50_dev, self.gamma_dev, self.offset_dev])
        self.parameters_fitted = [False, False, False]
        self.force_offset = force_offset
        self.aic = None
        self.bic = None

    def fit(self, X, y, d_50=None, gamma=None, offset=None, confidence_level=0.95, max_iter=1000):
        """Fits the provided data using iminuit's minimize function to minimize the negative log likelihood with regards
        to the parameters. Confidence intervals are computed through this package by looking at the profile likelihood
        and using the likelihood ratio test. Additional key-word parameters to the minimization function can be
        provided.

        Parameters:
            X: pandas.DataFrame
                The input data, should be a dataframe with at least two columns with both labels given in the object's
                initialization.
            y: array-like
                The target ground truth. This should be the known targets which depend on X.
            d_50: float or None
                If not None then this value will be used as d_50 and this parameter will not be fitted. Default is None.
            m: float or None
                If not None then this value will be used as m and this parameter will not be fitted. Default is None.
            n: float or None
                If not None then this value will be used as n and this parameter will not be fitted. Default is None.
            offset: float or None
                If not None then this value will be used as offset and this parameter will not be fitted. Default is
                None.
            confidence_level: float or None:
                The confidence level to be used in computing the parameter's confidence intervals.

        Returns:
            mle_model: iminuit.minuit.Minuit
                The output of the minimization function.
        """
        if X.shape[0] != y.shape[0]:
            raise AttributeError(f'size of input and target must match, got {X.shape[0]} and {y.shape[0]}')
        d = X[self.d_label].values
        nlle_log_reg = NLLLogReg(d, y, force_offset=self.force_offset)
        fixed_parameters = np.array([d_50, gamma, offset])
        self.parameters_fitted = fixed_parameters == None
        fixed_parameters = np.array(self.parameter_order)[~self.parameters_fitted]
        bounds = self._compute_bounds_(d_50, gamma, offset)
        for i in range(max_iter):
            try:
                d_50_, gamma_, offset_ = self._randomize_parameters_(d_50, gamma, offset)
                parameters = np.array([d_50_, gamma_, offset_])
                mle_model = Minuit(nlle_log_reg, parameters, name=nlle_log_reg.parameter_order)
                if fixed_parameters.size > 0:
                    mle_model.fixed[fixed_parameters] = True
                mle_model.limits = bounds
                mle_model.migrad()
                mle_model.minos(cl=confidence_level)
                break
            except RuntimeError as e:
                if i == max_iter-1:
                    raise e('Maximum number of iterations reached without fit succeeding.')
                else:
                    continue
        self.parameter_values = np.array([mle_model.params['d_50'].value, mle_model.params['gamma'].value,
                                          mle_model.params['offset'].value])
        self.parameter_devs = np.array([mle_model.params['d_50'].merror, mle_model.params['gamma'].merror,
                                        mle_model.params['offset'].merror], dtype=object)
        self.minimum_negative_log_likelihood_ = mle_model.fval
        self.maximum_likelihood_ = np.exp(-self.minimum_negative_log_likelihood_)
        self.aic = self._aic_()
        self.bic = self._bic_(X.shape[0])
        return mle_model

    def predict(self, X):
        """Computes the estimated output y following the probit model (LKB model) given the input data X and the fitted
        parameters. Should only be called after fit has been called.

        Parameters:
            X: pandas.DataFrame
                The input data, should be a dataframe with at least two columns with both labels given in the object's
                initialization.

        Returns:
            np.ndarray
                The corresponding output.
        """
        d = X[self.d_label].values
        return compute_log_reg(d, self.d_50(), self.gamma(), self.offset(), self.force_offset)

    def score(self, X, y, metric='auc', **kwargs):
        """Computes the score of the model in the given metric by analysing the predictions against the ground truth.

        Parameters:
            X: pandas.DataFrame
                The input data, should be a dataframe with at least two columns with both labels given in the object's
                initialization.
            y: array-like
                The target ground truth. This should be the known targets which depend on X.
            metric: str
                The metric to be used. Allowed values are 'auc'.

        Returns:
            score: np.ndarray
                The score.
        """
        probs = self.predict(X)
        if metric == 'auc':
            score = roc_auc_score(y, probs, **kwargs)
        return score

    def _aic_(self):
        """Computes the Akaike Information Criterion from the fitted model. Lower is better.

        Returns:
            float
                The AIC value.
        """
        # AIC = 2 * K - 2 ln(L), K is the number of fitted parameters in the model, L is the maximum value of likelihood
        aic = 2 * np.sum(self.parameters_fitted) + 2 * self.minimum_negative_log_likelihood_
        return aic

    def _bic_(self, n_patients):
        """Computes the Bayesian Information Criterion from the fitted model. Lower is better.

        Returns:
            float
                The BIC value.
        """
        # BIC = ln(n) * K - 2 ln(L), n is the number of patients, K is the number of fitted parameters in the model, L is the maximum value of likelihood
        bic = np.log(n_patients) * np.sum(self.parameters_fitted) + 2 * self.minimum_negative_log_likelihood_
        return bic

    @staticmethod
    def _randomize_parameters_(d_50=None, gamma=None, offset=None):
        """Randomly initializes parameters under each parameter limit. If a given parameter is not None then that
        parameter's value is not randomized.

        Parameters:
            d_50: float or None
                If not None this value will be used for d_50 and d_50 will not be randomized.
            gamma: float or None
                If not None this value will be used for gamma and gamma will not be randomized.
            offset: float or None
                If not None this value will be used for offset and offset will not be randomized.

        Returns:
            d_50: float
                The value for d_50
            gamma: float
                The value for gamma
            offset: float
                The value for offset
        """
        if d_50 is None:
            d_50 = np.random.random() * 10  # random number between [0,50)
        if gamma is None:
            gamma = np.random.normal(1, 0.3)
        if offset is None:
            offset = np.random.random()
        return d_50, gamma, offset

    def _compute_bounds_(self, d_50, gamma, offset):
        """Computes the limits for each parameter. Each parameter has specific physical limits in the LKB model. In
        case that parameter is fixed, then the limits should have that value as the lower and upper bound limits.

        Parameters:
            d_50: float or None
                The fixed value for d_50. If None then default Logistic Regression limits are used.
            gamma: float or None
                The fixed value for gamma. If None then default Logistic Regression limits are used.
            offset: float or None
                The fixed value for offset. If None then default Logistic Regression limits are used.

        Returns:
            np.ndarray
                An array of size four with three tuples giving the lower and upper bounds respectively for the parameter
                corresponding to that index. The parameter order can be checked from the LogReg object and it should be
                ['d_50', 'gamma', 'offset'].
        """
        bounds = self.bounds.copy()
        if d_50 is not None:
            bounds[0] = (d_50, d_50)
        if gamma is not None:
            bounds[1] = (gamma, gamma)
        if offset is not None:
            bounds[2] = (offset, offset)
        return np.array(bounds)

    def d_50(self):
        """Returns the d_50 value.

        Returns:
            float
                The d_50 value
        """
        return self.parameter_values[0]

    def gamma(self):
        """Returns the gamma value.

        Returns:
            float
                The gamma value
        """
        return self.parameter_values[1]

    def offset(self):
        """Returns the offset value.

        Returns:
            float
                The offset value.
        """
        return self.parameter_values[2]

    def d_50_dev(self):
        """Returns the d_50 deviance value.

        Returns:
            float
                The d_50 deviance value.
        """
        return self.parameter_devs[0]

    def gamma_dev(self):
        """Returns the gamma deviance value.

        Returns:
            float
                The gamma deviance value.
        """
        return self.parameter_devs[1]

    def offset_dev(self):
        """Returns the offset deviance value.

        Returns:
            float
                The offset deviance value.
        """
        return self.parameter_devs[2]


class NLLLogReg(object):
    """Callable to compute the negative log-likelihood associated with the LogisticRegression model. For physical
    context on the parameters, read the notes on the LogReg object or the compute_log_reg function.

    Parameters:
        d: np.ndarray
            The array holding the equivalent uniform dose or another dosimetric parameter.
        d_50: float or None
            If not None, this value will overwrite the entered d_50 value during call.
        gamma: float or None
            If not None, this value will overwrite the entered gamma value during call.
        offset: float or None
            If not None, this value will overwrite the entered offset value during call.
        force_offset: bool
            Modifier for the NTCP logistic function. Read `compute_log_reg` doc for more info.
    """
    def __init__(self, d, targets, d_50=None, gamma=None, offset=None, force_offset=False):
        self.d = d
        self.targets = targets
        self.parameter_order = ['d_50', 'gamma', 'offset']
        self.d_50 = d_50
        self.gamma = gamma
        self.offset = offset
        self.force_offset = force_offset
        self.fixed_parameters = [d_50, gamma, offset]
        self.errordef = iminuit.Minuit.LIKELIHOOD

    def __call__(self, parameters):
        """Computes the negative log-likelihood associated with the Logistic Regression model given the parameters.

        Parameters:
            parameters: array-like
                The parameters to be used in computation. Any parameters entered as fixed during initialization will be
                overwritten.

        Returns:
            neg_log_likelihood: float
                The computed value of the negative log likelihood.
        """
        all_params = np.array(self.fixed_parameters)
        all_params[all_params == None] = parameters
        d_50, gamma, offset = all_params
        scores = compute_log_reg(self.d, d_50, gamma, offset, self.force_offset)
        neg_log_likelihood = -1 * log_likelihood(scores, self.targets)
        return neg_log_likelihood

    def reset_parameters(self, d_50, gamma, offset):
        """Resets the fixed parameters that will overwrite those entered during call.

        Parameters:
            d_50: float or None
                The new fixed d_50 parameter. If None this parameter will stop being fixed.
            m: float or None
                The new fixed m parameter. If None this parameter will stop being fixed.
            n: float or None
                The new fixed n parameter. If None this parameter will stop being fixed.
            offset: float or None
                The new fixed offset parameter. If None this parameter will stop being fixed.
        """
        self.d_50 = d_50
        self.gamma = gamma
        self.offset = offset
        self.fixed_parameters = [d_50, gamma, offset]

    def copy(self):
        """Copies the current object

        Returns:
            complication_probability_models.NLLLKB
                The object copy
        """
        return NLLLogReg(self.d, self.targets, self.d_50, self.gamma, self.offset, self.force_offset)


def compute_log_reg(d, d_50, gamma, offset, force_offset=False):
    """Computes the logistic regression-based NTCP value for a given dosimetric parameter value `d` and parameters
    d_50, gamma, and offset.

    Parameters:
        d: float or array-like
            The value or values for which the NTCP is to be computed.
        d_50: float
            The d_50 value.
        gamma: float
            The gamma value.
        offset: float
            The offset value.
        force_offset: bool
            If True, the specified `offset` will be forced such that at d=0, NTCP=offset. This is done by translating
            the NTCP curve down so that NTCP_0=0 and then scaling such that it asymptotically approaches 1 still. The
            `offset` is added after this transformation of the NTCP curve such that after it NTCP_0=`offset`.

    Returns:
        float or array-like
            The NTCP value(s).
    """
    s = 4 * gamma / d_50
    ntcp = 1/(1+np.exp(s*(d_50-d)))
    if force_offset:
        ntcp_0 = 1/(1+np.exp(4 * gamma))
        ntcp = (ntcp - ntcp_0) / (1 - ntcp_0)
    return offset + ntcp * (1-offset)


def log_likelihood(scores, targets):
    """Computes the log-likelihood associated with the LKB model.

    Parameters:
        scores: np.ndarray
            The NTCP outputs of the model.
        targets: np.ndarray
            The ground truth (binary).

    Returns:
        float
            The log-likelihood.
    """
    log_ls = targets*np.log(scores + 1e-20)+(1 - targets)*np.log(1 - scores + 1e-20)
    return log_ls.sum()


def plot_fraction_better_vs_slope(data, x_column, y_column, conditions, legend_labels, colors, text_font='Calibri',
                                  title_size='11pt', text_size='10pt', **kwargs):
    plot = figure(
        sizing_mode="stretch_width",
        height=figure_height,
        max_width=figure_width,
        **kwargs
    )
    for i, row in conditions.iterrows():
        condition = data[row.index[0]] == row.iloc[0]
        for col, item in row.iteritems():
            condition = condition & (data[col] == item)
        x = data.loc[condition, x_column]
        y = data.loc[condition, y_column]
        plot.line(x, y, legend_label=legend_labels[i], line_color=colors[i])
        plot.circle(x, y, legend_label=legend_labels[i], line_color=colors[i], fill_color="white", size=6)
    plot.grid.grid_line_color = "white"
    plot.legend.location = "bottom_right"
    plot.legend.border_line_color = 'white'
    plot = edit_all_plot_text(plot, text_font, title_size, text_size)
    return plot


def plot_line_plus_std(data, x_column, y_column, conditions, legend_labels, colors, error_column=None,
                       error_type='whisker', text_font='Calibri', title_size='11pt', text_size='10pt', **kwargs):

    plot = figure(
        sizing_mode="stretch_width",
        height=figure_height,
        max_width=figure_width,
        **kwargs
    )
    y_min = 200
    y_max = -200
    for i, row in conditions.iterrows():
        condition = data[row.index[0]] == row.iloc[0]
        for col, item in row.iteritems():
            condition = condition & (data[col] == item)
        x = data.loc[condition, x_column]
        y = data.loc[condition, y_column]
        plot.line(x, y, legend_label=legend_labels[i], line_color=colors[i])
        plot.circle(x, y, legend_label=legend_labels[i], line_color=colors[i], fill_color="white", size=6)
        if error_column is not None:
            y_error = data.loc[condition, error_column]
            y_error_upper = (y + y_error).values
            y_error_lower = (y - y_error).values
            source = ColumnDataSource(data=dict(base=x, upper=y_error_upper, lower=y_error_lower))
            if error_type == 'whisker':
                error = Whisker(base='base', upper='upper', lower='lower', source=source, line_width=1,
                                line_color=colors[i], line_alpha=0.75)
                error.upper_head.line_color = colors[i]
                error.lower_head.line_color = colors[i]
                error.upper_head.line_alpha = 0.75
                error.lower_head.line_alpha = 0.75
            elif error_type == 'band':
                error = Band(base='base', upper='upper', lower='lower', source=source, fill_alpha=0.1,
                             fill_color=colors[i], line_color=colors[i], line_alpha=0.2, level='underlay')
        plot.add_layout(error)
        y_min = y_error_lower.min() if y_error_lower.min() < y_min else y_min
        y_max = y_error_upper.max() if y_error_upper.max() > y_max else y_max
    plot.grid.grid_line_color = "white"
    plot.legend.location = "bottom_right"
    plot.legend.border_line_color = 'white'
    plot = edit_all_plot_text(plot, text_font, title_size, text_size)
    buffer = (y_max-y_min)/10
    plot.y_range.start = y_min - buffer
    plot.y_range.end = y_max + buffer
    return plot


def plot_double_line_plus_std(data, x_column, y_column_1, y_column_2, conditions, legend_labels, colors,
                              y_2_error_column=None, y_2_axis_label=None, error_type='whisker', text_font='Calibri',
                              title_size='11pt', text_size='10pt', **kwargs):
    plot = figure(
        height=figure_height,
        width=figure_width + 150,
        **kwargs
    )
    auc_y_min = 200
    auc_y_max = -200

    for i, row in conditions.iterrows():
        condition = data[row.index[0]] == row.iloc[0]
        for col, item in row.iteritems():
            condition = condition & (data[col] == item)
        x = data.loc[condition, x_column]
        y_1 = data.loc[condition, y_column_1]
        y_2 = data.loc[condition, y_column_2]
        plot.line(x, y_1, legend_label=legend_labels[i][0], line_color=colors[i])
        plot.circle(x, y_1, legend_label=legend_labels[i][0], line_color=colors[i], fill_color="white", size=6)
        plot.line(x, y_2, legend_label=legend_labels[i][1], line_color=colors[i], line_dash='dotted',
                  y_range_name='second')
        plot.circle(x, y_2, legend_label=legend_labels[i][1], line_color=colors[i], fill_color="white", size=6,
                    y_range_name='second')
        if y_2_error_column is not None:
            y_error = data.loc[condition, y_2_error_column]
            y_error_upper = (y_2 + y_error).values
            y_error_lower = (y_2 - y_error).values
            source = ColumnDataSource(data=dict(base=x, upper=y_error_upper, lower=y_error_lower))
            if error_type == 'whisker':
                error = Whisker(base='base', upper='upper', lower='lower', source=source, line_width=1,
                                line_color=colors[i], line_alpha=0.75, y_range_name='second')
                error.upper_head.line_color = colors[i]
                error.lower_head.line_color = colors[i]
                error.upper_head.line_alpha = 0.75
                error.lower_head.line_alpha = 0.75
            elif error_type == 'band':
                error = Band(base='base', upper='upper', lower='lower', source=source, fill_alpha=0.1,
                             fill_color=colors[i], line_color=colors[i], line_alpha=0.2, line_dash='dotted',
                             level='underlay', y_range_name='second')
        plot.add_layout(error)
        auc_y_min = y_error_lower.min() if y_error_lower.min() < auc_y_min else auc_y_min
        auc_y_max = y_error_upper.max() if y_error_upper.max() > auc_y_max else auc_y_max
    plot.grid.grid_line_color = "white"
    plot.legend.location = "bottom_right"
    plot.legend.border_line_color = 'white'
    buffer = (auc_y_max-auc_y_min)/10
    plot.extra_y_ranges = {'second': Range1d(start=auc_y_min - buffer, end=auc_y_max + buffer)}
    plot.add_layout(LinearAxis(y_range_name='second', axis_label=y_2_axis_label), 'right')
    plot = edit_all_plot_text(plot, text_font, title_size, text_size)
    plot.add_layout(plot.legend[0], 'right')
    plot.width = plot.width + plot.legend[0].glyph_width
    return plot


def edit_all_plot_text(plot, text_font='Calibri', title_size='11pt', text_size='10pt'):
    plot.title.text_font = text_font
    plot.title.text_font_style = 'normal'
    plot.title.text_font_size = title_size
    plot.xaxis.major_label_text_font = text_font
    plot.xaxis.major_label_text_font_style = 'normal'
    plot.xaxis.major_label_text_font_size = text_size
    plot.xaxis.axis_label_text_font_style = 'normal'
    plot.xaxis.axis_label_text_font_size = text_size
    plot.yaxis.axis_label_text_font = text_font
    plot.yaxis.major_label_text_font = text_font
    plot.yaxis.major_label_text_font_style = 'normal'
    plot.yaxis.major_label_text_font_size = text_size
    plot.yaxis.axis_label_text_font_style = 'normal'
    plot.yaxis.axis_label_text_font_size = text_size
    plot.yaxis.axis_label_text_font = text_font
    if len(plot.legend) > 0:
        plot.legend.label_text_font = text_font
        plot.legend.label_text_font_size = text_size
        plot.legend.label_text_font_style = 'normal'
    return plot



if __name__ == '__main__':
    np.random.seed(0)
    # data_atlas = pd.read_parquet(r"\\umcfs013\RTHdata$\Research\Junior_Researcher_2019\NTCP_simulations\atlas.parquet")
    # # data_atlas = pd.read_csv(r"\\umcfs013\RTHdata$\Research\Junior_Researcher_2019\NTCP Comparison project\NKI_NTCP_comparison_data_Manual_Atlas_DL_Heart_ALL.csv")
    output_file(r'G:\Documents\Projects\NTCP project\René work\NTCP_simulations\bokeh.html')
    #
    # args = {
    #     'data': data_atlas.copy(),
    #     'model_organ': 'Heart_dosemean',
    #     'base_organ': 'Heart_dosemean',
    #     'test_organ': 'Heart_miguel_dosemean',
    #     # 'test_organ': 'Heart_atlas_dosemean',
    #     # 'test_organ': 'r',
    #     'parameters': {'d_50': (1.0, False), 'gamma': (1.0, False), 'offset': (0, True)},
    #     'n_iterations': 500,
    #     'fraction': 1,
    #     'ntcp_type': 'logistic',
    #     'maxfev_curvefit': 5000
    # }
    #
    # # np.random.seed(0) # for repeatability
    #
    # my_experiment = NTCPExperiment(**args)
    #
    # my_experiment.normalise_data()
    #
    # my_experiment.execute(calculate_bootstrap=True, parallel=True)
    #
    # my_experiment.plot()
    #
    # print(my_experiment.return_parameters())

    # results = pd.read_csv(r"G:\Documents\Projects\NTCP project\René work\NTCP_simulations\all_results.csv")
    # conditions = pd.DataFrame(
    #     [['Heart_dosemean', 'Heart_miguel_dosemean', 1], ['Heart_dosemean', 'Heart_atlas_dosemean', 1]],
    #     columns=['base_organ', 'organ', 'fraction_of_data'])
    # title = 'MHD: manual as the ground truth'
    # legends = ['AT: DL', 'AT: Atlas']
    # colors = ['navy', 'red']
    # x_axis_label = 'Gamma'
    # y_axis_label = 'Percentage Ground Truth Significantly Better'
    # plot = plot_fraction_better_vs_slope(results, 'slope', 'fraction_better', conditions, legends, colors, title=title,
    #                                      x_axis_label=x_axis_label, y_axis_label=y_axis_label, y_range=[0, 1.02],
    #                                      x_range=[0, 2.0])
    # # export_svg(plot, filename='MHD manual as the ground truth.svg')
    # show(plot)

    data_all = pd.read_csv(
        r"\\umcfs013\RTHdata$\Research\Junior_Researcher_2019\NTCP Comparison project\NKI_NTCP_comparison_data_Manual_Atlas_DL_Heart_ALL.csv")
    results = pd.read_csv(
        r"G:\Documents\Projects\NTCP project\René work\NTCP_simulations\long_simulation_04august_3000_it3_ground_truth.csv")

    y_axis_label = 'Fraction Ground Truth Significantly Better'
    y_axis_auc_label = 'Mean Pairwise AUC Difference'
    y_axis_aic_label = 'Mean Pairwise AIC Difference'
    gamma_axis_label = '\N{MATHEMATICAL ITALIC SMALL GAMMA}'
    gamma_x_range = [0, 1.57]
    y_range = [0, 1.05]
    v_axis_label = 'V (Gy)'

    colors = ['#3E6F89', '#BD481E', '#194734', '#EBA729', '#B098A4']

    np.random.seed(0)  # for repeatability
    conditions = pd.DataFrame(
        [['AutomaticMHD', 'ManualMHD', 1], ['AutomaticMHD', 'AtlasMHD', 1]],
        columns=['base_organ', 'organ', 'fraction_of_data'])
    title = 'MHD: DL as the ground truth'
    legends = ['AT: Manual', 'AT: Atlas']
    plot = plot_line_plus_std(results, 'slope', 'mean_pairwise_auc_difference', conditions, legends, colors[:2],
                              error_column='std_pairwise_auc_difference',
                              error_type='whisker', title=title, output_backend='svg', x_axis_label=gamma_axis_label,
                              y_range=y_range, x_range=gamma_x_range)
    # export_svg(plot, filename='MHD DL as the ground truth [Mean Pairwise AUC Difference].svg')
    show(plot)


