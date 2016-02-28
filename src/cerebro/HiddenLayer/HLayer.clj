(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

(defn output [inputs weights bias]
  (let [linear-output (reduce + (map * inputs weights))
        linear-output (+ linear-output bias)]
    (sigmoid linear-output)))

(defn hidden-sample-h|v [hidden-layer inputs]
  (let [{weights :W bias :bias} hidden-layer]
    (mapv (fn [w_i b_i] (binomial 1 (output inputs w_i b_i))) weights bias)))

(defn activation [layer input]
  (let [{W :weights bias :bias} layer
        activate (fn [inputs weights] (reduce + inputs weights))
        output   (map #(activate input %) W)
        bias-out (map + output bias)]
    (map sigmoid bias-out)))

