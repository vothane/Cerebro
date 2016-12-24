(ns cerebro.HiddenLayer.HLayer
  (:use [cerebro.Utils.utils]))

(defn HiddenLayer [w b]
  {:weights w
   :bias b

   :activation (fn [layer input]
                 (let [activate (fn [i w] (reduce + (map * i w)))
                       output   (map #(activate input %) :weights)
                       bias-out (map + output :bias)]
                   (map sigmoid bias-out)))
  })

(declare output)

(defn hidden-sample-h|v [hidden-layer inputs]
  (let [{weights :weights bias :bias} hidden-layer]
    (mapv (fn [w_i b_i] (binomial 1 (output inputs w_i b_i))) weights bias)))
     
    ;; helper functions 
    (defn output [inputs weights bias]
      (let [linear-output (reduce + (map * inputs weights))
            linear-output (+ linear-output bias)]
        (sigmoid linear-output)))

    