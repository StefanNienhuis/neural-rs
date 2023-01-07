use nalgebra::{DMatrix, DVector, Scalar};

fn to_vector<T: Scalar>(matrix: &DMatrix<T>) -> DVector<T> {
    return DVector::from_column_slice(matrix.data.as_slice());
}

fn to_matrix<T: Scalar>(shape: (usize, usize), vector: &DVector<T>) -> DMatrix<T> {
    return DMatrix::from_column_slice(shape.0, shape.1,vector.data.as_slice());
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector, matrix, vector};
    use crate::util::{to_vector, to_matrix};

    #[test]
    fn convert() {
        let vector = DVector::from_column_slice(&[1, 3, 2, 4]);
        let matrix = to_matrix((2, 2), &vector);

        assert_eq!(to_vector(&matrix), vector)
    }
}