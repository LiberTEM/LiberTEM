import { HandleRenderFunction } from "./types"

type ComposeFns = (f: HandleRenderFunction, g: HandleRenderFunction) => HandleRenderFunction;

export const composeHandles: ComposeFns = (f, g) => (handleDragStart, handleDrop) => (
    <>
        {f(handleDragStart, handleDrop)}
        {g(handleDragStart, handleDrop)}
    </>
)
