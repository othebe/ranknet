package neuralnet.activationfunction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IdentityActivationFunction implements IActivationFunction {
    public static final IdentityActivationFunction INSTANCE = new IdentityActivationFunction();

    private IdentityActivationFunction() { }

    public INDArray output(INDArray input) {
        return input;
    }

    public INDArray derivative(INDArray input) {
        return Nd4j.ones(input.shape());
    }
}
