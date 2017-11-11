

class IExecutionFramework(object):

    def get_kappas(self):
        """

        Returns
        -------
        pandas.Series<float>
            The cost of crossing trading the assets
        """
        raise NotImplementedError

