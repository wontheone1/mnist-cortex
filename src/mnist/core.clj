(ns mnist.core
  (:require
    [clojure.java.io :as io]
    [cortex.datasets.mnist :as mnist]
    [mikera.image.core :as i]
    [think.image.image :as image]
    [think.image.patch :as patch]
    [think.image.data-augmentation :as image-aug]
    [cortex.nn.layers :as layers]
    [clojure.core.matrix.macros :refer [c-for]]
    [clojure.core.matrix :as m]
    [cortex.experiment.classification :as classification]
    [cortex.experiment.train :as train]
    [cortex.nn.execute :as execute]
    [cortex.util :as util]
    [cortex.experiment.util :as experiment-util]))

(def image-size 28)

(def num-classes 10)

(def dataset-folder "mnist/")

(def training-dataset-folder (str dataset-folder "training"))

(def test-dataset-folder (str dataset-folder "test"))

(defn initial-description
  [input-w input-h num-classes]
  [(layers/input input-w input-h 1 :id :data)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization)
   (layers/linear 1000)
   (layers/relu :center-loss {:label-indexes        {:stream :labels}
                              :label-inverse-counts {:stream :labels}
                              :labels               {:stream :labels}
                              :alpha                0.9
                              :lambda               1e-4})
   (layers/dropout 0.5)
   (layers/linear num-classes)
   (layers/softmax :id :labels)])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Get Yann LeCun's mnist dataset and save it as folders of folders of png
;; files. The top level folders are named 'training' and 'test'. The subfolders
;; are named with class names, and those folders are filled with images of the
;; appropriate class.
(defn- ds-data->png
  "Given data from the original dataset, use think.image to produce png data."
  [ds-data]
  (let [data-bytes (byte-array (* image-size image-size))
        num-pixels (alength data-bytes)
        retval (image/new-image image/*default-image-impl*
                                image-size image-size :gray)]
    (c-for [idx 0 (< idx num-pixels) (inc idx)]
           (let [[x y] [(mod idx image-size)
                        (quot idx image-size)]]
             (aset data-bytes idx
                   (unchecked-byte (* 255.0
                                      (+ 0.5 (m/mget ds-data y x)))))))
    (image/array-> retval data-bytes)))

(defn- save-image!
  "Save a dataset image to disk."
  [output-dir [idx {:keys [data label]}]]
  (let [image-path (format "%s/%s/%s.png" output-dir (util/max-index label) idx)]
    (when-not (.exists (io/file image-path))
      (io/make-parents image-path)
      (i/save (ds-data->png data) image-path))
    nil))

(defn- timed-get-dataset
  [f name]
  (println "Loading" name "dataset.")
  (let [start-time (System/currentTimeMillis)
        ds (f)]
    (println (format "Done loading %s dataset in %ss"
                     name (/ (- (System/currentTimeMillis) start-time) 1000.0)))
    ds))

;; These two defonces use helpers from cortex to procure the original dataset.
(defonce training-dataset
         (timed-get-dataset mnist/training-dataset "mnist training"))

(defonce test-dataset
         (timed-get-dataset mnist/test-dataset "mnist test"))

(defonce ensure-images-on-disk!
         (memoize
           (fn []
             (println "Ensuring image data is built, and available on disk.")
             (dorun (map (partial save-image! training-dataset-folder)
                         (map-indexed vector training-dataset)))
             (dorun (map (partial save-image! test-dataset-folder)
                         (map-indexed vector test-dataset)))
             :done)))

(defn- image-aug-pipeline
  "Uses think.image augmentation to vary training inputs."
  [image]
  (let [max-image-rotation-degrees 25]
    (-> image
        (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                             max-image-rotation-degrees)
                          false)
        (image-aug/inject-noise (* 0.25 (rand))))))

(defn- mnist-observation->image
  "Creates a BufferedImage suitable for web display from the raw data
  that the net expects."
  [observation]
  (patch/patch->image observation image-size))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; The classification experiment system needs a way to go back and forth from
;; softmax indexes to string class names.
(def class-mapping
  {:class-name->index (zipmap (map str (range 10)) (range))
   :index->class-name (zipmap (range) (map str (range 10)))})

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Main entry point. In general, a classification experiment trains a net
;; forever, providing live updates on a local web server.
(defn train-forever
  ([] (train-forever {}))
  ([argmap]
   (ensure-images-on-disk!)
   (println "Training forever.")
   (let [train-ds (-> training-dataset-folder
                      (experiment-util/create-dataset-from-folder class-mapping
                                                                  :image-aug-fn (:image-aug-fn image-aug-pipeline))
                      (experiment-util/infinite-class-balanced-dataset))
         test-ds (-> test-dataset-folder
                     (experiment-util/create-dataset-from-folder class-mapping))
         listener (if-let [file-path (:tensorboard-output argmap)]
                    (classification/create-tensorboard-listener
                      {:file-path file-path})
                    (classification/create-listener mnist-observation->image
                                                    class-mapping
                                                    argmap))]
     (classification/perform-experiment
       (initial-description image-size image-size num-classes)
       train-ds test-ds listener))))

(def network-filename
  (str train/default-network-filestem ".nippy"))

(defn label-one
  "Take an arbitrary test image and label it."
  []
  (ensure-images-on-disk!)
  (let [observation (-> (str dataset-folder "test")
                        (experiment-util/create-dataset-from-folder class-mapping)
                        (rand-nth))]
    (i/show (mnist-observation->image (:data observation)))
    {:answer (-> observation :labels util/max-index)
     :guess  (->> (execute/run (util/read-nippy-file network-filename) [observation])
                  (first)
                  (:labels)
                  (util/max-index))}))
