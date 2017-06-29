package neuralnet.layer;

import neuralnet.activationfunction.IActivationFunction;
import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.weightinit.IWeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Builder {
    private int inCount;
    private int outCount;
    private IWeightInit weightInit;
    private IActivationFunction activationFunction = IdentityActivationFunction.INSTANCE;

    public Builder setInCount(int count) {
        this.inCount = count;

        return this;
    }

    public Builder setOutCount(int count) {
        this.outCount = count;

        return this;
    }

    public Builder setWeightInit(IWeightInit weightInit) {
        this.weightInit = weightInit;

        return this;
    }

    public Builder setActivationFunction(IActivationFunction activationFunction) {
        this.activationFunction = activationFunction;

        return this;
    }

    public Layer build() {
        INDArray weights = Nd4j.create(outCount, inCount).add(1);
        INDArray biases = Nd4j.create(1, outCount);

        Layer layer = new Layer(weights, biases, activationFunction);

        return layer;
    }
}
