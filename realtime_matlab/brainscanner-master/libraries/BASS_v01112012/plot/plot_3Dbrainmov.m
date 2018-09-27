function handles = plot_3Dbrain(vert,face,data11,data22,opts)
%==========================================================================
% Filename: plot_3Dbrain.m (function).
%
% Description:
%
% Input:        vert: Vertices
%               face: Faces
%               data: data (activity) that will be shown on brain
%               opts:
%                   .cMAP: colormap
%                   .taxis: Time axis in [s].
%                   .flag_interp: Do shading interpolation - default true.
%                   .flag_interactive: Allows interactive visualization
%                                      of the data using a sliderbar. Note
%                                      opts.taxis should also be specified.
%                                      Default: false
%                   .flag_colorbar: Show colorbar - default true.
%                   .flag_relval: Rescale to relative values with
%                                 max(abs(data)) as 1 - default false.
%                   .FaceAlpha: Transparency - default 1 no transparency
%
% Output:       handles: Handles to patch
%
% Example:      handles = plot_3Dbrain(vert,face,data)
%               handles = plot_3Dbrain(vert,face,data,opts)
%
% History:
%   - Created:  28/11/2008
%   - Modified: 27/12/2010: Now with slider to toogle through the source
%                           acitivity
%               22/01/2011: It is now possible to use the subplot function
%                           by using subplot outside this function.
%
% Copyright (C) Carsten Stahlhut, DTU Informatics 2011
%==========================================================================
startup_sub12
if (nargin<3 || isempty(data11))
    data = zeros(size(vert,1),1);
end

if nargin<4
    opts.dummy = [];
end

% try cMAP = opts.cMAP; catch cMAP = hsv(256); end;
try cMAP = opts.cMAP; catch cMAP = jet(256); end;
try crange = opts.crange; catch crange = [min(data(:)) max(data(:))]; end
try cBrain = opts.cBrain; catch cBrain = [0.7 0.7 0.7]; end
try fs = opts.fs; catch fs = 50; end;
try thresh = opts.thresh; catch thresh = 0.5; end
try taxis = opts.taxis; catch taxis = []; end;
try flag_interp = opts.flag_interp; catch flag_interp = true; end
try flag_interactive = opts.flag_interactive; catch flag_interactive = false; end
try flag_colorbar = opts.flag_colorbar; catch flag_colorbar = true; end
try hfig = opts.hfig; catch hfig = figure; end
try flag_relval = opts.flag_relval; catch flag_relval = false; end
try data_unit = opts.data_unit; catch data_unit = []; end
try FaceAlpha = opts.FaceAlpha; catch FaceAlpha = 1; end

if isfield(opts,'crangeSym'), crangeSym = opts.crangeSym; else crangeSym = false; end
if crangeSym
    crange = [-max(abs(crange)) max(abs(crange))];
end
if ~isfield(opts,'movie'), opts.movie = []; end

if isfield(opts.movie,'save'), flag_movie2file = opts.movie.save; else flag_movie2file=false; end
if isfield(opts.movie,'fname'), fname_movie = opts.movie.fname; else fname_movie = sprintf('movie_%s',date); end
if isfield(opts,'view'), view_coor = opts.view; else view_coor = [-168,12]; end
if isfield(opts,'figure_color'), figure_color = opts.figure_color; else figure_color = []; end
if isfield(opts,'text_color'), text_color = opts.text_color; else text_color = [0 0 0]; end
if isfield(opts,'flag_rotate'), flag_rotate = opts.flag_rotate; else flag_rotate = false; end
if isfield(opts,'view_drot'), view_drot = opts.view_drot; else view_drot = [0 0]; end



%% sub 1
data1=data11;data2=data22;
%data1(:,77:end)=repmat(data1(:,77),1,85);
%data2(:,77:end)=repmat(data2(:,77),1,85);
[Nd, Nt] = size(data1);


if ~isfield(opts,'cdata')
    
    if flag_relval
        data_max = max(abs(data(:)));
        data = data1/data_max;
    else 
        data = data1;
    end
    
    [data_max, t_max] = max( max( abs(data),[],1 ) );
    data_max
    % data = data(:,t_max);
    ithresh = find(abs(data(:)) <= data_max*thresh);
    clear t_max
    
    %% Color look up
    if diff(crange)~=0
        %Rescale data according to crange
        scale_out = length(cMAP);
        % range_out = [0 scale_out-1];
        range_out = [1 scale_out];      %Can ensure that activity should be zero to get the gray color
        cdata = rescaling(data,range_out,crange);
        cdata = cMAP(round(cdata(:)),:);
        %     keyboard
    else
        cdata = zeros(Nd,1,3);
    end
    cdata(ithresh,:) = repmat(cBrain,[length(ithresh) 1]);
    cdata = reshape(cdata,[Nd Nt 3]);
    cdata = permute(cdata,[1 3 2]);
    cdata(1)
else
    cdata = opts.cdata;     % Nd x 3 x Nt
end

handles.vert = vert;
handles.face = face;
handles.cdata1 = cdata;

handles.hfig = figure(hfig);
set(gcf,'Renderer','OpenGL')
if ~isempty(figure_color), set(gcf,'Color',figure_color); end
set(gca,'Visible','off')
subplot(2,2,1);title('-200 to 600 ms time window')
handles.patch1 = patch('vertices',handles.vert,...
    'faces',handles.face,'FaceVertexCData',handles.cdata1(:,:,1));
%subplot(2,1,2)
%handles.patch = patch('vertices',handles.vert,...
    %'faces',handles.face,'FaceVertexCData',handles.cdata(:,:,1));
colormap(cMAP)
if diff(crange)~=0, caxis(crange), else caxis([0 1]), end
set(handles.patch1,'FaceColor',cBrain,'EdgeColor','none','FaceAlpha',FaceAlpha);
if flag_interp, shading interp, else shading flat, end
lighting gouraud
camlight
zoom off
lightangle(0,270);lightangle(270,0),lightangle(0,0),lightangle(90,0);
material([.1 .1 .4 .5 .4]);

view(view_coor)
axis image
axis off
hold on
if flag_colorbar
    hcbar = colorbar;
    if ~isempty(data_unit)
        set(get(hcbar,'XLabel'),'String',data_unit)             %Colorbar label
    end
end
set(gcf,'Toolbar','figure')

%% sub2
clear opts.cdata
clear cdata
clear data
clear handles.cdata
clear handles.patch
if ~isfield(opts,'cdata')
    
    if flag_relval
        data_max = max(abs(data(:)));
        data = data2/data_max;
    else
        data = data2;
    end
    2
    [data_max, t_max] = max( max( abs(data),[],1 ) );
    data_max
    % data = data(:,t_max);
    ithresh = find(abs(data(:)) <= data_max*thresh);
    clear t_max
    
    %% Color look up
    if diff(crange)~=0
        %Rescale data according to crange
        scale_out = length(cMAP);
        % range_out = [0 scale_out-1];
        range_out = [1 scale_out];      %Can ensure that activity should be zero to get the gray color
        cdata = rescaling(data,range_out,crange);
        cdata = cMAP(round(cdata(:)),:);
        %     keyboard
    else
        cdata = zeros(Nd,1,3);
    end
    cdata(ithresh,:) = repmat(cBrain,[length(ithresh) 1]);
    cdata = reshape(cdata,[Nd Nt 3]);
    cdata = permute(cdata,[1 3 2]);
    cdata(1)
else
    cdata = opts.cdata;     % Nd x 3 x Nt
    1
end
handles.cdata2 = cdata;
subplot(2,2,3);title('100 to 200 ms time window')
handles.patch2 = patch('vertices',handles.vert,...
    'faces',handles.face,'FaceVertexCData',handles.cdata2(:,:,1));
%subplot(2,1,2)
%handles.patch = patch('vertices',handles.vert,...
    %'faces',handles.face,'FaceVertexCData',handles.cdata(:,:,1));
colormap(cMAP)
if diff(crange)~=0, caxis(crange), else caxis([0 1]), end
set(handles.patch2,'FaceColor',cBrain,'EdgeColor','none','FaceAlpha',FaceAlpha);
if flag_interp, shading interp, else shading flat, end
lighting gouraud
camlight
zoom off
lightangle(0,270);lightangle(270,0),lightangle(0,0),lightangle(90,0);
material([.1 .1 .4 .5 .4]);
annotation('arrow',[0.359333333333333 0.351753968253968],...
    [0.7183599677159 0.675972715998347],'LineWidth',3,'Color',[0 1 0]);
annotation('arrow',[0.165444444444444 0.180087301587301],...
    [0.721344987106046 0.67877820429536],'LineWidth',3,'Color',[0 0 1]);
annotation('arrow',[0.360333333333333 0.352753968253968],...
    [0.240290556900726 0.197903305183173],'LineWidth',3,'Color',[0 1 0]);
annotation('arrow',[0.168222222222222 0.182865079365079],...
    [0.244731589204512 0.202164806393826],'LineWidth',3,'Color',[0 0 1]);

view(view_coor)
axis image
axis off

hold on
if flag_colorbar
    hcbar = colorbar;
    if ~isempty(data_unit)
        set(get(hcbar,'XLabel'),'String',data_unit)             %Colorbar label
    end
end
set(gcf,'Toolbar','figure')
% set(handles.fig1,'facecolor','w');

%---------------------------------------------------------------------
% Multiple time points show movie or toogle through the data using
% interactive plotting
%---------------------------------------------------------------------
if flag_interactive
    if ~isempty(taxis)
        minorStep = 1/(length(taxis)-1);
        majorStep = minorStep*5;
        handles.taxis = taxis;
        
        handles.slideText = uicontrol('Style','text',...
            'Position',[440 20 75 20],...
            'String',[num2str(round(taxis(1)*1e3)) ' ms']);
        
        handles.slidebar = uicontrol('Style', 'slider',...
            'Min',1,'Max',length(taxis),'Value',1,...
            'Position', [125 20 300 20],...
            'SliderStep',[minorStep majorStep],...
            'Callback', {@sliderUpdate,handles});
    else
        error('Chosing interactive plot requires the field taxis to be specified in opts.')
    end
else
    
    if flag_movie2file
        
        aviobj = avifile([fname_movie '.avi'],'fps',5);
        fig1=handles.hfig;
        winsize = get(fig1,'Position'); winsize(1:2) = [0 0];
        %         aviobj = avifile('movie.avi','fps',5);
        numframes=16;
        set(fig1,'Renderer','zbuffer');
        
        
        if ~isempty(taxis)
            % Create textbox
            ha1 = annotation(handles.hfig,'textbox',...
                [0.382222222222222 0.00179624441307068 0.256666666666667 0.0664652567975831],...
                'String',{sprintf('Time: %3.3f sec',taxis(1) ) },...
                'FitBoxToText','on','Color',text_color);
            
            for ti=1:size(data,2)
                
                if flag_rotate
                    view(view_coor(1),view_coor(2))
                    view_coor = view_coor + view_drot;
                    if abs(view_coor(2))>=90
                        view_drot(2) = -view_drot(2);
                        view_coor(2) = view_coor(2) + 2*view_drot(2);
                    end
                end
                
                set(ha1,'String',{sprintf('Time: %3.3f sec',taxis(ti) ) } )
                set(handles.patch1,'FaceVertexCdata',handles.cdata1(:,:,ti));
                set(handles.patch2,'FaceVertexCdata',handles.cdata2(:,:,ti));
                                timet = linspace(-0.2,0.6-1/fs,161);

                subplot(2,2,2);plot(timet(1:ti),data11(105,1:ti),timet(1:ti),data11(4287,1:ti),'g'); % plot command 
                set(gcf,'Color',[1 1 1]);
                xlim([-0.205 0.6]);ylim([-0.15 1.35]);
                xlabel('Time [s]'); ylabel('Source strength');legend('Strongest source','2nd strongest source','Location','NorthWest')
                 subplot(2,2,4);d2 = ones(8196,161)*NaN;d2(:,61:80)=data22(:,61:80);
                 plot(timet(1:ti),d2(105,1:ti),timet(1:ti),d2(4287,1:ti),'g'); % plot command 
                set(gcf,'Color',[1 1 1]);
                xlim([-0.205 0.6]);ylim([-0.15 1.35]);
                xlabel('Time [s]'); ylabel('Source strength');legend('Strongest source','2nd strongest source','Location','NorthWest')
                     
                drawnow
                frame = getframe(fig1,winsize);
                aviobj = addframe(aviobj,frame);
                A(:,ti) = frame;
                
                pause(1/(fs))
            end
            
            %         axis tight;
            %         set(gca,'NextPlot','replaceChildren');
            
        end
        aviobj = close(aviobj);
        save movie.mat A
        
    else
        
        if ~isempty(taxis)
            % Create textbox
            ha1 = annotation(handles.hfig,'textbox',...
                [0.74 0.017 0.21 0.067],...
                'String',{sprintf('Time: %3.3f sec',taxis(1) ) },...
                'FitBoxToText','off');
            
            for ti=1:size(data,2)
                set(ha1,'String',{sprintf('Time: %3.3f sec',taxis(ti) ) } )
                set(handles.patch,'FaceVertexCdata',handles.cdata(:,:,ti));
                drawnow
                pause(1/(fs))
            end
            
        else
            if size(data,2)>1
                ha1 = annotation(handles.hfig,'textbox',...
                    [0.74 0.017 0.21 0.067],...
                    'String',{sprintf('Time: %3.3f sec',1 ) },...
                    'FitBoxToText','off');
                
                for ti=1:size(data,2)
                    set(ha1,'String',{sprintf('Sample: %3.0f',ti ) } )
                    set(handles.patch,'FaceVertexCdata',handles.cdata(:,:,ti));
                    drawnow
                    pause(1/(fs))
                end
            end
        end
        
    end
    
    
end

end


function sliderUpdate(hObj,event,info)
%==========================================================================
% Filename: sliderUpdate.m (function).
%
% Description:  Called to set FaceVertexCdata of patch in figure axes
%               when user moves the slider control
%
% Input:        hObj: Handle to the slider object.
%               event: Faces
%               info: handles with info
%
% Output:       None
%
% Copyright (C) Carsten Stahlhut, DTU Informatics 2011
%==========================================================================
ti = round(get(hObj,'Value'));
set(info.patch,'FaceVertexCdata',info.cdata(:,:,ti));
set(info.slideText,'String',[num2str(round(info.taxis(ti)*1e3)) ' ms'])
end