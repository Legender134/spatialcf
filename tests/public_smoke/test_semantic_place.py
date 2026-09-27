"""Installed M5 CPU smoke; fresh solve and checker on each invocation."""
from tests.public_smoke._placement_fixtures import exact, transform, rectangle, placement_scene, table_cavity

def run_placement_case(operation='PLACE_ON',backend_enabled=True):
    from spatialcf.core.semantic_place_compiler import build_semantic_place_request
    from spatialcf.core.semantic_place_backend import SemanticPlaceBackend
    from spatialcf.core.outcome_assembler import assemble_counterfactual_outcome,assemble_no_selection_unknown
    from spatialcf.domain.semantic_place import SemanticPlaceInterval
    root=build_semantic_place_request(scene=placement_scene(),subject_id='entity:subject',operation=operation,
        target_ids=('cavity:table',) if operation=='PLACE_IN' else ('surface:table',),
        x=SemanticPlaceInterval(lower=-10.0,upper=10.0),y=SemanticPlaceInterval(lower=-10.0,upper=10.0),
        z=SemanticPlaceInterval(lower=0.0,upper=10.0),cavities=(table_cavity(),),backend_enabled=backend_enabled)
    backend=SemanticPlaceBackend()
    selection=backend.select(root)
    if selection.selection_disposition=='NO_SELECTION':
        assembled=assemble_no_selection_unknown(solve_request=root,selection=selection)
    else:
        compiled=backend.compile(root)
        submission=backend.solve_submission(compiled,root.solver_config)
        assembled=assemble_counterfactual_outcome(solve_request=root,selection=selection,compilation=compiled,submission=submission)
    assert assembled.result.structural_outcome_class==('CERTIFIED_SOLUTION' if backend_enabled else 'UNKNOWN')
    if backend_enabled:
        subject=next(o for o in assembled.program.after_scene_state.base_scene_payload.objects.values if o.object_id=='entity:subject')
        assert subject.support_assignment.surface_id=='surface:table'
        xyz=subject.pose.world_from_object.translation
        assert (xyz.x,xyz.y,xyz.z)==(3.5,0.0,1.5)
    else:
        assert assembled.checked_proof_outcome is None and assembled.certificate is None
        assert all(entry.used==0.0 for entry in assembled.result.resource_usage.entries)
    return root.solve_request_sha256,assembled.result.solve_result_sha256


def test_installed_place_on():
    run_placement_case('PLACE_ON')


def test_installed_place_in():
    run_placement_case('PLACE_IN')


def test_installed_place_no_selection():
    run_placement_case(backend_enabled=False)
