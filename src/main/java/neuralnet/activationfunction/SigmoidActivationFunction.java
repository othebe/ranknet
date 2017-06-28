package neuralnet.activationfunction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class SigmoidActivationFunction implements IActivationFunction {
    public static final IActivationFunction INSTANCE = new SigmoidActivationFunction();

    public INDArray output(INDArray input) {
        return Transforms.sigmoid(input);
    }

    public INDArray derivative(INDArray input) {
        INDArray sigmoid = Transforms.sigmoid(input);
        INDArray ones = Nd4j.ones(sigmoid.shape());

        return sigmoid.mul(ones.sub(sigmoid));
    }
}
