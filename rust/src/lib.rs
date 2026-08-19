use pyo3::prelude::*;

mod hyp_functions;
mod seth_functions;

pub use hyp_functions::{
    HypergraphError, HypergraphRustic, MorphismRustic, cardinality_indexed_by_links,
    cardinality_indexed_by_nodes, decode_hypergraph_morphism_by_nodes, homgraph_cardinality_fast,
};

#[pymodule]
fn rustic(m: &Bound<'_, PyModule>) -> PyResult<()> {
    seth_functions::register(m)?;
    hyp_functions::register(m)?;

    Ok(())
}
