(ns cerebro.Utils.utils)

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn binomial [n p]
  (let [c 0]
    (if (or (< p 0) (> p 1))
      c
      (for [_ (range n) :let [r (rand)]]
        (if (< r p) (inc c) c)))))

(defn dot [v w] (reduce + (map * v w)))

(defn vector-transpose [v] (mapv vector v))

(defn matrix-transpose [m]
  (apply mapv vector m))