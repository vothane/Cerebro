(ns cerebro.Utils.utils)

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn binomial [n p]
    (let [c 0]
          (if (or (< p 0) (> p 1))
            c
            (loop [iter n acc c]
              (if (= iter 0)
                acc
                (recur (dec iter) (if (< (rand) p) (inc acc) acc)))))))

(defn vector-transpose [v] (mapv vector v))

(defn matrix-transpose [m]
  (apply mapv vector m))
