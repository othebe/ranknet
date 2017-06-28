package neuralnet.activationfunction;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IActivationFunction {
    INDArray output(INDArray input);
    INDArray derivative(INDArray input);
}
