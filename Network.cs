using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork {
    /// <summary>
    ///     Represents a Feed Forward <see cref="Network" /> of <see cref="Neuron" />.
    /// </summary>
    internal class Network {
        private static readonly IEnumerable<double> Bias = new List<double> {1};
        private readonly IEnumerable<Layer> _layers;

        /// <summary>
        ///     Create an instance of a single instance of Feed Forward neural <see cref="Network" />.
        /// </summary>
        /// <param name="size">
        ///     Size of the <see cref="Network" />.
        ///     First element is number of inputs and last element is number of outputs.
        /// </param>
        /// <param name="learningRate">Rate of 1 is normal rate, 0 is no rate. Default to 0.1</param>
        public Network(IReadOnlyCollection<int> size, double learningRate = 0.1) =>
            _layers = Enumerable
                .Range(0, size.Count - 1)
                .Select((_, i) => new Layer(size.ElementAt(i + 1), size.ElementAt(i) + 1, learningRate))
                .ToList();

        /// <summary>
        ///     Predict the output by feeding the output of each <see cref="Layer" />
        ///     to the input of the next <see cref="Layer" />.
        /// </summary>
        /// <param name="input">Sequence of input values.</param>
        /// <returns>Predicted output.</returns>
        public IEnumerable<double> Predict(IEnumerable<double> input) =>
            _layers.Aggregate(input, (inputArr, layer) => layer.Predict(Bias.Concat(inputArr)));

        /// <summary>
        ///     Make the <see cref="Network" /> learn using the difference between predicted and expected output.
        ///     The difference is fed from the last <see cref="Layer" /> to the first <see cref="Layer" />.
        ///     Finally the weight of each <see cref="Neuron" /> is updated.
        /// </summary>
        /// <param name="predicted">Sequence of double values. Usually output of <see cref="Predict" />.</param>
        /// <param name="expected">
        ///     Sequence of double values.
        ///     Usually what the output of <see cref="Predict" /> should have been.
        /// </param>
        public void Learn(IEnumerable<double> predicted, IEnumerable<double> expected) {
            var errors = Subtract(predicted, expected);
            _layers.Reverse().Aggregate(errors, (error, layer) => layer.BackPropagation(error));
            _layers.ToList().ForEach(layer => layer.Update());
        }

        private static IEnumerable<double> Subtract(IEnumerable<double> arr1, IEnumerable<double> arr2) =>
            arr1.Select((e, i) => e - arr2.ElementAt(i));
    }
}