import neuralnet.NeuralNet;
import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.activationfunction.SigmoidActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class Regression {
    private static int FEATURE_COUNT = 20;
    private static int HIDDEN_NODE_COUNT = 40;
    private static int EPOCHS = 20;

    public static void main(String[] args) {
        NeuralNet net = NeuralNet.Builder()
                .setLearningRate(.01)
                .addLayer(Layer.Builder().setInCount(FEATURE_COUNT).setOutCount(HIDDEN_NODE_COUNT).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(HIDDEN_NODE_COUNT).setOutCount(HIDDEN_NODE_COUNT).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(HIDDEN_NODE_COUNT).setOutCount(HIDDEN_NODE_COUNT).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(HIDDEN_NODE_COUNT).setOutCount(1).setActivationFunction(IdentityActivationFunction.INSTANCE).build())
                .build();

        INDArray testInput = Nd4j.rand(1, FEATURE_COUNT);

        List<INDArray> outputs = net.feedForward(testInput);
        System.out.printf("FeedForward: %s\n\n", outputs.toString());

        List<INDArray> gradients = net.backpropagateError(Nd4j.create(new double[] { 2 }, new int[] { 1, 1 }));
        System.out.printf("ErrorGradient: %s\n\n", gradients.toString());

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            List<INDArray> activations = net.feedForward(testInput);
            double output = activations.get(activations.size() - 1).getDouble(0);
            System.out.printf("Epoch %d : %f\n", epoch, output);

            net.train(testInput, Nd4j.create(new double[] { 12 }, new int[] { 1, 1 }));
        }
    }
}
