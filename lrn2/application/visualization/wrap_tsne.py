import tsne

class tSNE_bh:
    """
    t-Distributed Stochastic Neighbor Embedding using the Barnes Hut algorithm.
    Uses tsne package

    Parameters
    ----------

    observations: array_like
        Matrix of observations

    M: integer
        Target dimensionality. Default is 2

    perplexity: float
        Effective number of neighbors. Default is 30.

    theta: float
        Trade off factor. If smaller the accuracy is better, but it takes considerably longer to make the embedding.
        Default 0.5

    pca_d: integer, optional
        Preprocess the data using randomized PCA. This setes the target dimensionality for PCA.
        Default is None

    Returns
    -------

    .observations: numpy array
        Input observations, or rPCA transformed observations

    .dimensionality
        Input target dimensionality

    .perp
        Input perplexity

    .theta
        Input trade off factor

    Methods
    -------

    tsne_transform
        Computes the embedding
    """

    def __init__(self,observations,M=2,perplexity=30.0,theta=0.5,pca_d=None):
        # Preprocess the data
        if pca_d is None:
            self.observations = observations
        else:
            tmp = rPCA(observations,pca_d)
            self.observations = tmp.pca_transform()

        self.dimensionality = M
        self.perp = perplexity
        self.theta = theta

    def tsne_transform(self):
        """
        Computes the dimensionality reduction

        Returns
        -------

        bhsne: numpy array
            Matrix of transformed data
        """
        return tsne.bh_sne(self.observations,
                      d=self.dimensionality,
                      perplexity=self.perp,
                      theta=self.theta)

