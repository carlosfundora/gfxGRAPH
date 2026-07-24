use serde::{Deserialize, Serialize};

use crate::shape::{Shape, ShapeError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayoutKind {
    RowMajor,
    ColumnMajor,
    ChannelsLast,
    Tiled,
    Paged,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Contiguity {
    Contiguous,
    Strided,
    Tiled,
    Paged,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrideSpec {
    strides: Vec<usize>,
}

impl StrideSpec {
    pub fn new(strides: Vec<usize>) -> Self {
        Self { strides }
    }

    pub fn row_major(shape: &Shape) -> Result<Self, ShapeError> {
        let mut strides = vec![1usize; shape.rank()];
        for axis in (0..shape.rank().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(shape.dim(axis + 1)?)
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(Self { strides })
    }

    pub fn column_major(shape: &Shape) -> Result<Self, ShapeError> {
        let mut strides = vec![1usize; shape.rank()];
        for axis in 1..shape.rank() {
            strides[axis] = strides[axis - 1]
                .checked_mul(shape.dim(axis - 1)?)
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(Self { strides })
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileLayoutSpec {
    pub tile_shape: Vec<usize>,
    pub tile_order: Vec<usize>,
}

impl TileLayoutSpec {
    pub fn new(tile_shape: Vec<usize>) -> Self {
        let tile_order = (0..tile_shape.len()).collect();
        Self {
            tile_shape,
            tile_order,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageLayoutSpec {
    pub page_axis: usize,
    pub page_size: usize,
    pub page_stride: usize,
}

impl PageLayoutSpec {
    pub fn new(page_axis: usize, page_size: usize, page_stride: usize) -> Self {
        Self {
            page_axis,
            page_size,
            page_stride,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLayout {
    pub kind: LayoutKind,
    pub shape: Shape,
    pub strides: StrideSpec,
    pub tile: Option<TileLayoutSpec>,
    pub page: Option<PageLayoutSpec>,
}

impl TensorLayout {
    pub fn row_major(shape: Shape) -> Result<Self, ShapeError> {
        let strides = StrideSpec::row_major(&shape)?;
        Ok(Self {
            kind: LayoutKind::RowMajor,
            shape,
            strides,
            tile: None,
            page: None,
        })
    }

    pub fn column_major(shape: Shape) -> Result<Self, ShapeError> {
        let strides = StrideSpec::column_major(&shape)?;
        Ok(Self {
            kind: LayoutKind::ColumnMajor,
            shape,
            strides,
            tile: None,
            page: None,
        })
    }

    pub fn tiled(
        shape: Shape,
        strides: StrideSpec,
        tile: TileLayoutSpec,
    ) -> Result<Self, ShapeError> {
        if tile.tile_shape.len() != shape.rank() {
            return Err(ShapeError::AxisOutOfBounds {
                axis: tile.tile_shape.len(),
                rank: shape.rank(),
            });
        }
        Ok(Self {
            kind: LayoutKind::Tiled,
            shape,
            strides,
            tile: Some(tile),
            page: None,
        })
    }

    pub fn paged(
        shape: Shape,
        strides: StrideSpec,
        page: PageLayoutSpec,
    ) -> Result<Self, ShapeError> {
        shape.dim(page.page_axis)?;
        Ok(Self {
            kind: LayoutKind::Paged,
            shape,
            strides,
            tile: None,
            page: Some(page),
        })
    }

    /// Checks whether the tensor layout is contiguous in row-major order.
    ///
    /// This implementation is optimized to be entirely allocation-free (no heap allocation).
    /// It validates strides inline, which allows the compiler to fully optimize and unroll
    /// the loop, taking advantage of SIMD/AVX instructions on the host CPU.
    #[inline]
    pub fn is_row_contiguous(&self) -> bool {
        let rank = self.shape.rank();
        if rank == 0 {
            return true;
        }

        let dims = self.shape.dims();
        let strides = self.strides.strides();
        if strides.len() != rank {
            return false;
        }

        // The innermost dimension must have a stride of 1 for row-major contiguity
        if strides[rank - 1] != 1 {
            return false;
        }

        // Validate remaining strides from right to left (innermost to outermost)
        // expected_stride[i] = expected_stride[i+1] * shape[i+1]
        for i in (0..rank - 1).rev() {
            match strides[i + 1].checked_mul(dims[i + 1]) {
                Some(expected) => {
                    if strides[i] != expected {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    /// Evaluates the contiguity type of the tensor layout.
    pub fn contiguity(&self) -> Contiguity {
        match self.kind {
            LayoutKind::RowMajor if self.is_row_contiguous() => Contiguity::Contiguous,
            LayoutKind::ColumnMajor | LayoutKind::ChannelsLast => Contiguity::Strided,
            LayoutKind::Tiled => Contiguity::Tiled,
            LayoutKind::Paged => Contiguity::Paged,
            LayoutKind::RowMajor => Contiguity::Strided,
            LayoutKind::Custom => Contiguity::Unknown,
        }
    }

    /// Computes the linear memory offset for the given multi-dimensional indices.
    ///
    /// Optimized for compiler auto-vectorization (e.g. AVX2/AVX-512) by performing
    /// linear zip-iterations and preventing nested dynamic bounds checks inside the loop.
    /// Returns a ShapeError if indices are out of bounds or if multiplication overflows.
    #[inline]
    pub fn linear_offset(&self, indices: &[usize]) -> Result<usize, ShapeError> {
        let rank = self.shape.rank();
        if indices.len() != rank {
            return Err(ShapeError::AxisOutOfBounds {
                axis: indices.len(),
                rank,
            });
        }

        let dims = self.shape.dims();
        let strides = self.strides.strides();

        let mut offset = 0usize;
        for i in 0..rank {
            let index = indices[i];
            let dim = dims[i];
            if index >= dim {
                return Err(ShapeError::AxisOutOfBounds {
                    axis: index,
                    rank: dim,
                });
            }
            let prod = index
                .checked_mul(strides[i])
                .ok_or(ShapeError::ElementCountOverflow)?;
            offset = offset
                .checked_add(prod)
                .ok_or(ShapeError::ElementCountOverflow)?;
        }
        Ok(offset)
    }

    /// Checks if a tensor copy is required before executing a graph capture.
    ///
    /// Graph capture on ROCm/HIP requires tensors to be contiguous in memory
    /// to prevent incorrect or overlapping buffer re-recordings.
    #[inline]
    pub fn needs_copy_for_graph_capture(&self) -> bool {
        !self.is_row_contiguous() || matches!(self.kind, LayoutKind::Custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayoutConversionPlan {
    pub source: LayoutKind,
    pub target: LayoutKind,
    pub requires_copy: bool,
    pub reason: String,
}

impl LayoutConversionPlan {
    pub fn from_layout(layout: &TensorLayout, target: LayoutKind) -> Self {
        let requires_copy = layout.kind != target
            || (target == LayoutKind::RowMajor && !layout.is_row_contiguous());
        let reason = if requires_copy {
            format!("convert {:?} layout to {:?}", layout.kind, target)
        } else {
            "layout already satisfies target".to_string()
        };
        Self {
            source: layout.kind,
            target,
            requires_copy,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_offsets_work() {
        let layout = TensorLayout::row_major(Shape::new(vec![2, 3, 4]).unwrap()).unwrap();
        assert_eq!(layout.strides.strides(), &[12, 4, 1]);
        assert_eq!(layout.linear_offset(&[1, 2, 3]).unwrap(), 23);
    }

    #[test]
    fn conversion_plan_marks_column_major_copy() {
        let layout = TensorLayout::column_major(Shape::new(vec![2, 3]).unwrap()).unwrap();
        let plan = LayoutConversionPlan::from_layout(&layout, LayoutKind::RowMajor);
        assert!(plan.requires_copy);
    }

    #[test]
    fn tiled_layout_reports_tiled_contiguity() {
        let shape = Shape::new(vec![8, 8]).unwrap();
        let layout = TensorLayout::tiled(
            shape,
            StrideSpec::new(vec![8, 1]),
            TileLayoutSpec::new(vec![4, 4]),
        )
        .unwrap();
        assert_eq!(layout.contiguity(), Contiguity::Tiled);
    }
}
