
import pandas as pd


class IStrategy(object):
    """
    IStrategyFactory

    This is the interface for all strategies that we can add to our model
    """

    def get_spin_up_period(self, freq):
        """
        All strategies operate over a frequency spectrum. The strategy can
        return its required spin up period for the given frequency.

        Parameters
        ----------
        freq: float
            The frequency we are aiming to be trading at

        Returns
        -------
        float
            The multiple of freq we require for our spin up
        """
        raise NotImplementedError

    def portfolio_update(self, portfolio):
        """
        When the portfolio of positions managed by this strategy change the
        strategy will be notified with this call

        Parameters
        ----------
        portfolio: pandas.Series

        """
        raise NotImplementedError

    def handle_price_update(self, prices):
        """

        Parameters
        ----------
        prices: pandas.DataFrame # TODO - replace with a different class
            Prices of full universe of assets

        Returns
        -------
        pandas.Series
            Portfolio of desired positions
        """
        raise NotImplementedError


class IStrategyFactory(object):

    def create(self, low_frequency, high_frequency, asset_universe,
               tradeable_assets, execution_framework):
        """
        Create the implementation of the strategy for the given frequencies

        Parameters
        ----------
        low_frequency: float
        high_frequency: float
        asset_universe: list<str>
        tradeable_assets: list<str>
            A subset of the asset universe
        execution_framework: execution.IExecutionFramework

        Returns
        -------
        IStrategy
        """
        raise NotImplementedError


class MVOStrategy(IStrategy):

    def __init__(self, portfolio, strategy_detail, execution_framework):
        self._strategy_detail = strategy_detail
        self._execution_framework = execution_framework
        self._portfolio = portfolio

    def get_spin_up_period(self, freq):
        return self._strategy_detail.get_spin_up_period(freq)

    def portfolio_update(self, portfolio):
        self._portfolio = portfolio

    def handle_price_update(self, prices):
        self._strategy_detail.handle_price_update(prices)
        alphas = self._strategy_detail.get_current_alphas()
        cov_matrix = self._strategy_detail.get_current_cov_matrix()
        kappas = self._execution_framework.get_costs()

        return self._do_mvo(alphas, cov_matrix, kappas)

    def _do_mvo(self, alphas, cov_matrix, kappas):
        pass


class Correlation(object):

    def __init__(self, asset_universe, tradeable_assets):
        self._betas = pd.DataFrame(0.0, columns=asset_universe,
                                   index=tradeable_assets)
        self._log_returns = None

    @property
    def betas(self):
        return self._betas

    def get_spin_up_period(self, freq):
        return 50  # TODO - ???

    def handle_price_update(self, prices):
        self._log_returns = prices.log_returns
        self.update_betas()

    def get_alphas(self):
        pass

    def get_cov_matrix(self):
        pass

    def update_betas(self):
        pass
    