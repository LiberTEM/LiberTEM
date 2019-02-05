import DraggableHandle from "./DraggableHandle";

export type HandleDragStartFn = (h: DraggableHandle) => void;
export type HandleDropFn = () => void

export type HandleRenderFunction = (
    handleDragStart: HandleDragStartFn,
    handleDrop: HandleDropFn,
    onKeyboardEvent?: (e: React.KeyboardEvent<SVGElement>) => void,
) => (JSX.Element | null)