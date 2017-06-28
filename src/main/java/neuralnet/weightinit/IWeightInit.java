package neuralnet.weightinit;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IWeightInit {
    // Convert an array of 1's.
    INDArray init(INDArray weights);
}
