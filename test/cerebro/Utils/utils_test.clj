(ns cerebro.Utils.utils_test
  (:require [clojure.test :refer :all]
            [cerebro.Utils.utils :refer :all]))

(deftest dot-product-test
  (testing "dot product matrix operation"
    (is (= (dot-product [1 3 -5] [4 -2 -1]) 3))))

(deftest transpose-test
  (testing "transpose matrix operation"
    (is (= (transpose [[1 2 3] [4 5 6]]) [[1 4] [2 5] [3 6]]))))