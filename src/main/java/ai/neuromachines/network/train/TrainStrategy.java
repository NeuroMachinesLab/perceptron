package ai.neuromachines.network.train;

import ai.neuromachines.network.Network;

public interface TrainStrategy {

    static TrainStrategy backpropagation(Network network) {
        return BackpropagationTrainStrategy.of(network);
    }

    /**
     * Sets network input signal to given one and trains network weights for expected output signal
     *
     * @see <a href="https://en.wikipedia.org/wiki/Backpropagation">Backpropagation</a>
     */
    void train(float[] input, float[] expectedOutput);

    void setLearningRate(float learningRate);

    float getLearningRate();
}
