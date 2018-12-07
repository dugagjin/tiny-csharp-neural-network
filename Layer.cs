using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork {
    /// <summary>
    ///     Represents a <see cref="Layer" /> which manages a sequence of <see cref="Neuron" />.
    ///     It can be used standalone or in a <see cref="Network" />.
    /// </summary>
    internal class Layer {
        private readonly IEnumerable<Neuron> _neurons;

        /// <summary>
        ///     Create an instance of a single <see cref="Layer" /> which
        ///     consists of a <see cref="Neuron" /> sequence.
        /// </summary>
        /// <param name="size">Number of <see cref="Neuron" /> in the <see cref="Layer" />.</param>
        /// <param name="inputs">Number of input data.</param>
        /// <param name="learningRate">Rate of 1 is normal rate, 0 is no rate.</param>
        public Layer(int size, int inputs, double learningRate) =>
            _neurons = Enumerable.Range(0, size).Select(_ => new Neuron(inputs, learningRate)).ToList();

        /// <summary>
        ///     Compute the output of each <see cref="Neuron" /> according the input.
        /// </summary>
        /// <param name="input">Sequence of input values.</param>
        /// <returns>Predicted output.</returns>
        public IEnumerable<double> Predict(IEnumerable<double> input) =>
            _neurons.Select(neuron => neuron.Predict(input));

        /// <summary>
        ///     For each <see cref="Neuron" />, each weight, except bias, is multiplied with the error.
        ///     The result is reduced by addition.
        /// </summary>
        /// <param name="errors">Difference between predicted and expected output.</param>
        /// <returns>Sum of the weight scaled with the error.</returns>
        public IEnumerable<double> BackPropagation(IEnumerable<double> errors) =>
            _neurons.Select((neuron, i) => neuron.BackPropagation(errors.ElementAt(i))).Aggregate(Add);

        /// <summary>
        ///     Update the weights of each <see cref="Neuron" /> according to the input, output, error and learning rate.
        /// </summary>
        public void Update() => _neurons.ToList().ForEach(neuron => neuron.Update());

        private static IEnumerable<double> Add(IEnumerable<double> arr1, IEnumerable<double> arr2) =>
            arr1.Select((e, i) => e + arr2.ElementAt(i));
    }
}