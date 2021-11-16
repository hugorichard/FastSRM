(defun hello (name)
  (insert "Hello " name "!\n"))
(setq list-of-names '("Sarah" "Chloe" "Mathilde"))

(defun whosname ()
  (let ((your-name (read-from-minibuffer "Enter your name: ")))
    (switch-to-buffer-other-window "*test*")
    (erase-buffer)
    (insert (format "Hello %s" your-name))
    (other-window 1)
   ))

(defun hello-to-bonjour ()
  (switch-to-buffer-other-window "*test*")
  (erase-buffer)
  ;; Say hello to names in `list-of-names'
  (mapcar 'hello list-of-names)
  (goto-char (point-min))
  ;; Replace "Hello" by "Bonjour"
  (while (search-forward "Hello" nil t)
    (replace-match "Bonjour"))
  (other-window 1))

(hello-to-bonjour)

(setq list-of-names '("Sarah" "Chloe" "Mathilde"))
(defun boldify-names ()
  (switch-to-buffer-other-window "*test*")
  (goto-char (point-min))
  (while (re-search-forward "Bonjour \\(.+\\)!" nil t)
    (add-text-properties (match-beginning 1)
                         (match-end 1)
                         (list 'face 'bold)))
  (other-window 1))

(boldify-names)
