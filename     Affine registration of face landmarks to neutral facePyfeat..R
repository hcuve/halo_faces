# Neutral face coordinates
neutral = pd.read_csv(
  os.path.join(get_resource_path(), "neutral_face_coordinates.csv"), index_col=False
)


def registration(face_lms, neutral=neutral, method="fullface"):
  """Register faces to a neutral face.

    Affine registration of face landmarks to neutral face.

    Args:
        face_lms(array): face landmarks to register with shape (n,136). Columns 0~67 are x coordinates and 68~136 are y coordinates
        neutral(array): target neutral face array that face_lm will be registered
        method(str or list): If string, register to all landmarks ('fullface', default), or inner parts of face nose,mouth,eyes, and brows ('inner'). If list, pass landmarks to register to e.g. [27, 28, 29, 30, 36, 39, 42, 45]

    Return:
        registered_lms: registered landmarks in shape (n,136)
    """
assert type(face_lms) == np.ndarray, TypeError("face_lms must be type np.ndarray")
assert face_lms.ndim == 2, ValueError("face_lms must be shape (n, 136)")
assert face_lms.shape[1] == 136, ValueError("Must have 136 landmarks")
registered_lms = []
for row in face_lms:
  face = [row[:68], row[68:]]
face = np.array(face).T
#   Rotate face
primary = np.array(face)
secondary = np.array(neutral)
n = primary.shape[0]
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:, :-1]
X1, Y1 = pad(primary), pad(secondary)
if type(method) == str:
  if method == "fullface":
  A, res, rank, s = np.linalg.lstsq(X1, Y1, rcond=None)
elif method == "inner":
  A, res, rank, s = np.linalg.lstsq(X1[17:, :], Y1[17:, :], rcond=None)
else:
  raise ValueError("method is either 'fullface' or 'inner'")
elif type(method) == list:
  A, res, rank, s = np.linalg.lstsq(X1[method], Y1[method], rcond=None)
else:
  raise TypeError(
    "method is string ('fullface','inner') or list of landmarks"
  )
transform = lambda x: unpad(np.dot(pad(x), A))
registered_lms.append(transform(primary).T.reshape(1, 136).ravel())
return np.array(registered_lms)