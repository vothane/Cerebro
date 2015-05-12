(ns cerebro.Utils.utils)

(defn dot-product [& matrix]
  (apply + (apply map * matrix)))

(defmulti transpose class)
 
(defmethod transpose clojure.lang.PersistentList
  [matrix]
  (apply map list matrix))
 
(defmethod transpose clojure.lang.PersistentVector
  [matrix]
  (vec (apply map vector matrix)))

