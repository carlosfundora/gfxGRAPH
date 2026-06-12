use serde::{Deserialize, Serialize};

use rs_gfxgraph_core::{
    BucketRouterCore, BucketState, DTypeConversionContract, GraphCapabilityRegistry,
    HardwareProfile, KernelLaunchShape, LayoutConversionPlan, LayoutKind, Shape, ShapeError,
    ShapeKey, ShapeLayoutConversionPlan, SignalError, SpectralShape, TensorLayout, WaveError,
    WindowKind, WindowSpec, WorkgroupShape,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeBucketDecision {
    pub axis: usize,
    pub input: usize,
    pub bucket: usize,
    pub state: BucketState,
    pub bucketed_shape: Shape,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphShapeReport {
    pub shape_key: ShapeKey,
    pub element_count: usize,
    pub rank: usize,
    pub row_contiguous: bool,
    pub graph_capture_copy_required: bool,
    pub conversion_plan: LayoutConversionPlan,
    pub full_conversion_plan: ShapeLayoutConversionPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlan {
    pub hardware: HardwareProfile,
    pub launch: KernelLaunchShape,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolboxError {
    Shape(String),
    Router(String),
    Signal(String),
    Wave(String),
}

impl std::fmt::Display for ToolboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolboxError::Shape(message) => write!(f, "shape error: {message}"),
            ToolboxError::Router(message) => write!(f, "router error: {message}"),
            ToolboxError::Signal(message) => write!(f, "signal error: {message}"),
            ToolboxError::Wave(message) => write!(f, "wave error: {message}"),
        }
    }
}

impl std::error::Error for ToolboxError {}

impl From<ShapeError> for ToolboxError {
    fn from(value: ShapeError) -> Self {
        ToolboxError::Shape(value.to_string())
    }
}

impl From<WaveError> for ToolboxError {
    fn from(value: WaveError) -> Self {
        ToolboxError::Wave(value.to_string())
    }
}

impl From<SignalError> for ToolboxError {
    fn from(value: SignalError) -> Self {
        ToolboxError::Signal(value.to_string())
    }
}

pub struct GfxGraphToolbox;

impl GfxGraphToolbox {
    pub fn bucket_shape_axis(
        shape: &Shape,
        axis: usize,
        buckets: Vec<usize>,
    ) -> Result<ShapeBucketDecision, ToolboxError> {
        let input = shape.dim(axis)?;
        let router = BucketRouterCore::new(buckets);
        let (bucket, state) = router
            .route(input)
            .map_err(|err| ToolboxError::Router(err.to_string()))?;
        Ok(ShapeBucketDecision {
            axis,
            input,
            bucket,
            state,
            bucketed_shape: shape.ceil_axis_to(axis, bucket)?,
        })
    }

    pub fn analyze_shape_for_graph_capture(
        operation: impl Into<String>,
        dtype: impl Into<String>,
        backend: impl Into<String>,
        layout: TensorLayout,
        target_layout: LayoutKind,
    ) -> Result<GraphShapeReport, ToolboxError> {
        let conversion_plan = LayoutConversionPlan::from_layout(&layout, target_layout);
        let full_conversion_plan =
            ShapeLayoutConversionPlan::plan(&layout, layout.shape.clone(), target_layout, None)?;
        let shape_key = ShapeKey::new(
            operation,
            dtype,
            &layout.shape,
            format!("{:?}", layout.kind).to_ascii_lowercase(),
            backend,
        );
        Ok(GraphShapeReport {
            shape_key,
            element_count: layout.shape.element_count()?,
            rank: layout.shape.rank(),
            row_contiguous: layout.is_row_contiguous(),
            graph_capture_copy_required: layout.needs_copy_for_graph_capture(),
            conversion_plan,
            full_conversion_plan,
        })
    }

    pub fn plan_dtype_shape_layout_conversion(
        layout: &TensorLayout,
        target_shape: Shape,
        target_layout: LayoutKind,
        dtype_contract: DTypeConversionContract,
    ) -> Result<ShapeLayoutConversionPlan, ToolboxError> {
        Ok(ShapeLayoutConversionPlan::plan(
            layout,
            target_shape,
            target_layout,
            Some(dtype_contract),
        )?)
    }

    pub fn capability_registry() -> GraphCapabilityRegistry {
        GraphCapabilityRegistry::baseline()
    }

    pub fn rdna2_decode_launch(blocks: u32) -> Result<LaunchPlan, ToolboxError> {
        let hardware = HardwareProfile::rdna2_gfx1030();
        let launch = KernelLaunchShape {
            grid: WorkgroupShape::new(blocks.max(1), 1, 1)?,
            block: hardware.recommended_decode_block,
        };
        launch.validate(hardware.wavefront)?;
        Ok(LaunchPlan { hardware, launch })
    }

    pub fn build_window(spec: WindowSpec) -> Result<Vec<f32>, ToolboxError> {
        if spec.length == 0 {
            return Err(SignalError::ZeroLength.into());
        }

        match spec.kind {
            WindowKind::Rectangular => Ok(vec![1.0; spec.length]),
            WindowKind::Hann => Ok(hann_window(spec.length, spec.periodic)),
        }
    }

    pub fn frequency_bins(shape: SpectralShape) -> Result<Vec<f32>, ToolboxError> {
        (0..shape.bins)
            .map(|bin| shape.hz_for_bin(bin).map_err(ToolboxError::from))
            .collect()
    }
}

fn hann_window(length: usize, periodic: bool) -> Vec<f32> {
    if length == 1 {
        return vec![1.0];
    }

    let denominator = if periodic { length } else { length - 1 } as f32;
    (0..length)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / denominator;
            0.5 - 0.5 * phase.cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toolbox_buckets_shape_axis() {
        let shape = Shape::new(vec![2, 127, 64]).unwrap();
        let decision = GfxGraphToolbox::bucket_shape_axis(&shape, 1, vec![64, 128, 256]).unwrap();
        assert_eq!(decision.bucket, 128);
        assert_eq!(decision.bucketed_shape.dims(), &[2, 128, 64]);
    }

    #[test]
    fn toolbox_reports_capture_layout() {
        let layout = TensorLayout::row_major(Shape::new(vec![2, 4]).unwrap()).unwrap();
        let report = GfxGraphToolbox::analyze_shape_for_graph_capture(
            "decode",
            "fp16",
            "rdna2",
            layout,
            LayoutKind::RowMajor,
        )
        .unwrap();
        assert!(report.row_contiguous);
        assert!(!report.graph_capture_copy_required);
        assert!(!report.full_conversion_plan.requires_copy);
        assert_eq!(report.element_count, 8);
    }

    #[test]
    fn toolbox_exposes_capability_registry() {
        assert!(GfxGraphToolbox::capability_registry().has("convert.shape_layout_dtype"));
    }

    #[test]
    fn toolbox_builds_rdna2_launch() {
        let plan = GfxGraphToolbox::rdna2_decode_launch(4).unwrap();
        assert_eq!(plan.hardware.name, "rdna2-gfx1030");
        assert_eq!(plan.launch.block.threads(), 32);
    }

    #[test]
    fn toolbox_builds_hann_window() {
        let spec = WindowSpec::new(WindowKind::Hann, 4).unwrap().symmetric();
        let window = GfxGraphToolbox::build_window(spec).unwrap();
        assert_eq!(window.len(), 4);
        assert!(window[0].abs() < f32::EPSILON);
        assert!(window[3].abs() < f32::EPSILON);
    }

    #[test]
    fn toolbox_builds_frequency_bins() {
        let bins = GfxGraphToolbox::frequency_bins(SpectralShape::new(8_000, 8).unwrap()).unwrap();
        assert_eq!(bins, vec![0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0]);
    }
}
