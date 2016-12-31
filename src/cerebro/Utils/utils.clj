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

(defn dot-product [& matrix]
  {:pre [(apply == (map count matrix))]}
  (apply + (apply map * matrix)))

(defprotocol Matrix
  (el [m i j])
  (put [matrix i j f])
  (range-rows [m])
  (range-cols [m])
  (size [m]))

(extend-protocol Matrix
  clojure.lang.IPersistentVector
  (el [m i j]
    (get-in m [i j]))
  (put [m i j f]
    (assoc-in m [i j] (f m i j)))
  (range-rows [m] 
    (range (count m)))
  (range-cols [m] 
    (range (count (first m))))
  (size [m] 
    [(count m) (count (first m))]))