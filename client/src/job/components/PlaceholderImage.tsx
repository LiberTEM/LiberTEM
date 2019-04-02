import { ReactElement } from "react";
import * as React from "react";
import styled from 'styled-components';

interface PlaceholderProps {
    width: number,
    height: number,
}

interface AspectPaddingProps {
    aspect: number,
}

const AspectPadding = styled.div`
    padding-bottom: ${(props: AspectPaddingProps) => `${props.aspect}%`},
    width: 100%;
    position: relative;
`;

const PlaceholderImage: React.SFC<PlaceholderProps> = ({ children, width, height }) => {
    const aspect = 100 * (height / width);
    return (
        <AspectPadding aspect={aspect}>
            {
                React.Children.map(children, child => {
                    if (!React.isValidElement(child)) {
                        return child;
                    }
                    return React.cloneElement(child as ReactElement<any>, {
                        style: {
                            position: "absolute",
                            left: 0,
                            top: 0,
                            ...(child.props as any).style,
                        },
                    });
                })
            }
        </AspectPadding>
    )
}

export default PlaceholderImage;