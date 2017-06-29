import neuralnet.NeuralNet;
import neuralnet.activationfunction.IdentityActivationFunction;
import neuralnet.activationfunction.SigmoidActivationFunction;
import neuralnet.layer.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;

public class Regression {
    public static void main(String[] args) {
        NeuralNet net = NeuralNet.Builder()
                .setLearningRate(.001)
                .addLayer(Layer.Builder().setInCount(2).setOutCount(4).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(4).setOutCount(4).setActivationFunction(SigmoidActivationFunction.INSTANCE).build())
                .addLayer(Layer.Builder().setInCount(4).setOutCount(1).build())
                .build();

        INDArray testInput = Nd4j.create(new double[] { 2, 3 }, new int[] { 2, 1 });

        List<INDArray> outputs = net.feedForward(testInput);
        System.out.println(outputs);

        List<INDArray> gradients = net.backpropagateError(Nd4j.scalar(2));
        System.out.println(gradients);

        for (int iteration = 0; iteration < 10; iteration++) {
            List<INDArray> activations = net.feedForward(testInput);
            double output = activations.get(activations.size() - 1).getDouble(0);
            System.out.printf("Iteration %d : %f\n", iteration, output);

            net.train(testInput, Nd4j.scalar(12));
        }
    }
}
