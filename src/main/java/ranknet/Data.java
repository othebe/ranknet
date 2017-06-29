package ranknet;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Data {
    private double rankScore;
    private INDArray features;

    private double computeScore;

    public Data(double rankScore, INDArray features, double computeScore) {
        this.rankScore = rankScore;
        this.features = features;

        this.computeScore = computeScore;
    }

    public double getRankScore() {
        return rankScore;
    }

    public INDArray getFeatures() {
        return features;
    }

    public double getComputeScore() {
        return computeScore;
    }
}
