(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

(defn output [inputs weights bias]
  (let [linear-output (reduce + (map * inputs weights))
        linear-output (+ linear-output bias)]
    (sigmoid linear-output)))

(defn sample-h-given-v [hidden-layer biases inputs]
  (mapv #(binomial 1 (output inputs %1 %2)) hidden-layer biases))
