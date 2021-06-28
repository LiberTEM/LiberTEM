import React from "react";

export const useDismissEscape = (dismiss: () => void): void => {
    React.useEffect(() => {
        const handleEsc = (ev: KeyboardEvent) => {
            if(ev.code === "Escape" || ev.keyCode === 27) {
                dismiss();
            }
        }
        document.addEventListener("keyup", handleEsc);

        return () => {
            document.removeEventListener("keyup", handleEsc);
        };
    });
}