(ns cerebro.Utils.utils_test
  (:use clojure.core.matrix)
  (:require [clojure.test :refer :all]
            [cerebro.Utils.utils :refer :all]))

(deftest dot-product-test
  (testing "dot product on two matrices"
    (is (= (dp [[1 0] [0 1]] [[4 1] [2 2]]) [[4 1] [2 2]]))))