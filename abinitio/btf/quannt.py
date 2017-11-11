

class IQuannt(object):
    pass


class Quannt(IQuannt):

    def __init__(self, strategy_factories, universe):
        """

        Parameters
        ----------
        strategy_factories: list<strategy.IStrategyFactory>
        """
        self._strategy_factories = strategy_factories
        self._universe = universe

    def learn(self, data_generator, tradeable_assets, freq,
              total_batches=1000000):

        architecture = self.prepare_architecture(freq, tradeable_assets)

        while self._batches_seen < total_batches:
            self.learn_batch(architecture, data_generator)

    def prepare_architecture(self, freq, tradeable_assets):
        pass


# get start time
# get end time
# generate portfolio position
# run backtest for each strategy
    # p_i := portfolio(strategy_i)
    # w_i := final node output
    # portfolio = w_i * p_i
    # U(portfolio)
# Update the net
